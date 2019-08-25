from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events, Engine
from ignite.metrics import Loss
from ignite.contrib.handlers import LRScheduler
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import torch
import copy


class LRFinder:
    """Learning rate range test.

    The learning rate range test increases the learning rate in a pre-training run
    between two boundaries in a linear or exponential manner. It provides valuable
    information on how well the network can be trained over a range of learning rates
    and what is the optimal learning rate.

    Arguments:
        model (torch.nn.Module): wrapped model.
        optimizer (torch.optim.Optimizer): wrapped optimizer where the defined learning
            is assumed to be the lower boundary of the range test.
        criterion (torch.nn.Module): wrapped loss function.
        device (str or torch.device, optional): a string ("cpu" or "cuda") with an
            optional ordinal for the device type (e.g. "cuda:X", where is the ordinal).
            Alternatively, can be an object representing the device on which the
            computation will take place. Default: None, uses the same device as `model`.
        memory_cache (boolean): if this flag is set to True, `state_dict` of model and
            optimizer will be cached in memory. Otherwise, they will be saved to files
            under the `cache_dir`.
        cache_dir (string): path for storing temporary files. If no path is specified,
            system-wide temporary directory is used.
            Notice that this parameter will be ignored if `memory_cache` is True.

    Example:
        '>>> lr_finder = LRFinder(net, optimizer, criterion, device="cuda")'
        '>>> lr_finder.range_test(dataloader, end_lr=100, num_iter=100)'

    Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    fastai/lr_find: https://github.com/fastai/fastai

    """

    def __init__(self, model, optimizer, criterion, device=None, memory_cache=True, cache_dir=None, verbose=False):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.history = {"lr": [], "loss": []}
        self.best_loss = None
        self.memory_cache = memory_cache
        self.cache_dir = cache_dir

        # Save the original state of the model and optimizer so they can be restored if
        # needed
        self.model_device = next(self.model.parameters()).device
        self.state_cacher = StateCacher(memory_cache, cache_dir=cache_dir)
        self.state_cacher.store('model', self.model.state_dict())
        self.state_cacher.store('optimizer', self.optimizer.state_dict())

        # If device is None, use the same as the model
        if device:
            self.device = device
        else:
            self.device = self.model_device

        self.lr_suggestion = None

        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.DEBUG)

    def reset(self):
        """Restores the model and optimizer to their initial states."""
        self.model.load_state_dict(self.state_cacher.retrieve('model'))
        self.optimizer.load_state_dict(self.state_cacher.retrieve('optimizer'))
        self.model.to(self.model_device)
        self.logger.info("restored model and optimizer to initial state")

    def range_test(self, train_loader, val_loader=None, end_lr=10, num_iter=100, step_mode="exp", smooth_f=0.05,
                   diverge_th=5, suggestion=True):
        """Performs the learning rate range test.

        Arguments:
            train_loader (torch.utils.data.DataLoader): the training set data laoder.
            val_loader (torch.utils.data.DataLoader, optional): if `None` the range test
                will only use the training loss. When given a data loader, the model is
                evaluated after each iteration on that dataset and the evaluation loss
                is used. Note that in this mode the test takes significantly longer but
                generally produces more precise results. Default: None.
            end_lr (float, optional): the maximum learning rate to test. Default: 10.
            num_iter (int, optional): the number of iterations over which the test
                occurs. Default: 100.
            step_mode (str, optional): one of the available learning rate policies,
                linear or exponential ("linear", "exp"). Default: "exp".
            smooth_f (float, optional): the loss smoothing factor within the [0, 1[
                interval. Disabled if set to 0, otherwise the loss is smoothed using
                exponential smoothing. Default: 0.05.
            diverge_th (int, optional): the test is stopped when the loss surpasses the
                threshold:  diverge_th * best_loss. Default: 5.
            suggestion (bool, optional): whether to compute suggested learning rate (minimal grad) and store value into
                {lr_finder_name}.lr_suggestion. Default: True

        """

        self.logger.info("Learning rate search started")
        # Reset test results
        self.history = {"lr": [], "loss": []}
        self.best_loss = None

        # Initialize the proper learning rate policy
        if step_mode.lower() == "exp":
            lr_schedule = LRScheduler(ExponentialLR(self.optimizer, end_lr, num_iter))
        elif step_mode.lower() == "linear":
            lr_schedule = LRScheduler(LinearLR(self.optimizer, end_lr, num_iter))
        else:
            raise ValueError(f"expected one of (exp, linear), got {step_mode}")

        if smooth_f < 0 or smooth_f >= 1:
            raise ValueError("smooth_f is outside the range [0, 1]")

        trainer = create_supervised_trainer(self.model, self.optimizer, self.criterion, self.device, non_blocking=True)

        # if val_loader provided, calculates average loss across entire validation set, accurate but very very slow
        if val_loader:
            evaluator = create_supervised_evaluator(self.model, metrics={"Loss": Loss(self.criterion)},
                                                    device=self.device, non_blocking=True)
            trainer.add_event_handler(Events.ITERATION_COMPLETED, lambda engine: evaluator.run(val_loader))

        # log the loss at the end of every train iteration

        def log_lr_and_loss(finder):
            loss = evaluator.state.metrics["Loss"] if val_loader else trainer.state.output
            lr = lr_schedule.lr_scheduler.get_lr()[0]
            finder.history["lr"].append(lr)
            if trainer.state.iteration == 1:
                finder.best_loss = loss
            else:
                if smooth_f > 0:
                    loss = smooth_f * loss + (1 - smooth_f) * finder.history["loss"][-1]
                if loss < finder.best_loss:
                    finder.best_loss = loss
            finder.history["loss"].append(loss)

        trainer.add_event_handler(Events.ITERATION_COMPLETED, lambda engine: log_lr_and_loss(self))

        # increase lr with every iteration
        trainer.add_event_handler(Events.ITERATION_COMPLETED, lr_schedule)

        # Check if the loss has diverged; if it has, stop the trainer
        def loss_diverged(engine: Engine, finder):
            if finder.history["loss"][-1] > diverge_th * finder.best_loss:
                engine.terminate()
                finder.logger.info("Stopping early, the loss has diverged")

        trainer.add_event_handler(Events.ITERATION_COMPLETED, lambda engine: loss_diverged(engine, self))

        # run lr finder
        trainer.run(train_loader, 999)

        if suggestion:
            self.lr_suggestion = self._suggestion()

        self.logger.info("Learning rate search finished. See the graph with {finder_name}.plot()")

    def plot(self, skip_start=10, skip_end=5, log_lr=True):
        """Plots the learning rate range test.

        Arguments:
            skip_start (int, optional): number of batches to trim from the start.
                Default: 10.
            skip_end (int, optional): number of batches to trim from the start.
                Default: 5.
            log_lr (bool, optional): True to plot the learning rate in a logarithmic
                scale; otherwise, plotted in a linear scale. Default: True.

        """

        if skip_start < 0:
            raise ValueError("skip_start cannot be negative")
        if skip_end < 0:
            raise ValueError("skip_end cannot be negative")

        # Get the data to plot from the history dictionary. Also, handle skip_end=0
        # properly so the behaviour is the expected
        lrs = self.history["lr"]
        losses = self.history["loss"]
        if skip_end == 0:
            lrs = lrs[skip_start:]
            losses = losses[skip_start:]
        else:
            lrs = lrs[skip_start:-skip_end]
            losses = losses[skip_start:-skip_end]

        # Plot loss as a function of the learning rate
        plt.plot(lrs, losses)
        if log_lr:
            plt.xscale("log")
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        plt.show()

    def _suggestion(self):
        """
        Returns: learning rate at the minimum numerical gradient
        """
        loss = self.history["loss"]
        grads = [loss[i] - loss[i-1] for i in range(1, len(loss))]
        min_grad_idx = np.argmin(grads) + 1
        return self.history["lr"][int(min_grad_idx)]


class LinearLR(_LRScheduler):
    """Linearly increases the learning rate between two boundaries over a number of
    iterations.

    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float, optional): the initial learning rate which is the lower
            boundary of the test. Default: 10.
        num_iter (int, optional): the number of iterations over which the test
            occurs. Default: 100.
        last_epoch (int): the index of last epoch. Default: -1.

    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]


class ExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a number of
    iterations.

    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float, optional): the initial learning rate which is the lower
            boundary of the test. Default: 10.
        num_iter (int, optional): the number of iterations over which the test
            occurs. Default: 100.
        last_epoch (int): the index of last epoch. Default: -1.

    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


class StateCacher(object):

    def __init__(self, in_memory, cache_dir=None):
        self.in_memory = in_memory
        self.cache_dir = cache_dir

        if self.cache_dir is None:
            import tempfile
            self.cache_dir = tempfile.gettempdir()
        else:
            if not os.path.isdir(self.cache_dir):
                raise ValueError('Given `cache_dir` is not a valid directory.')

        self.cached = {}

    def store(self, key, state_dict):
        if self.in_memory:
            self.cached.update({key: copy.deepcopy(state_dict)})
        else:
            fn = os.path.join(self.cache_dir, 'state_{}_{}.pt'.format(key, id(self)))
            self.cached.update({key: fn})
            torch.save(state_dict, fn)

    def retrieve(self, key):
        if key not in self.cached:
            raise KeyError('Target {} was not cached.'.format(key))

        if self.in_memory:
            return self.cached.get(key)
        else:
            fn = self.cached.get(key)
            if not os.path.exists(fn):
                raise RuntimeError('Failed to load state in {}. File does not exist anymore.'.format(fn))
            state_dict = torch.load(fn, map_location=lambda storage, location: storage)
            return state_dict

    def __del__(self):
        """Check whether there are unused cached files existing in `cache_dir` before
        this instance being destroyed."""
        if self.in_memory:
            return

        for k in self.cached:
            if os.path.exists(self.cached[k]):
                os.remove(self.cached[k])
