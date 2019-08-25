from super_convergence.one_cycle import OneCycleLR
from super_convergence.lr_finder_engine import LRFinder
from super_convergence.radam import RAdam
from torch.optim.optimizer import Optimizer
from ignite.engine import Engine, Events, create_supervised_trainer, _prepare_batch
from ignite.contrib.handlers import LRScheduler
import logging


def super_convergence_engine(model, criterion, optimizer=None, device=None, non_blocking=False,
                             prepare_batch=_prepare_batch, output_transform=lambda x, y, y_pred, loss: loss.item(),
                             engine_create_func=None):

    logger = logging.getLogger(__name__)
    if optimizer is None:
        logger.info("received None for 'optimizer', inits RAdam optimizer")
        optimizer = RAdam(model.parameters(), lr=1e-6)

    lr_finder = LRFinder(model, optimizer, criterion, device=device)

    if engine_create_func is None:
        trainer = create_supervised_trainer(model, optimizer, criterion, device=device, non_blocking=non_blocking,
                                            prepare_batch=prepare_batch, output_transform=output_transform)
    else:
        trainer = engine_create_func(model, optimizer, criterion, device=device, non_blocking=non_blocking,
                                     prepare_batch=prepare_batch, output_transform=output_transform)

    trainer.add_event_handler(Events.STARTED, find_lr_add_one_cycle, lr_finder, optimizer)

    return trainer


def find_lr_add_one_cycle(engine: Engine, lr_finder: LRFinder, optimizer: Optimizer):
    train_dl = engine.state.dataloader
    lr_finder.range_test(train_dl, num_iter=1000)
    max_lr = lr_finder.lr_suggestion
    lr_finder.reset()
    one_cycle_scheduler = LRScheduler(OneCycleLR(optimizer, max_lr, train_dl=train_dl,
                                                 num_epochs=engine.state.max_epochs))
    engine.add_event_handler(Events.ITERATION_STARTED, one_cycle_scheduler)
