import os
import io
from setuptools import setup, find_packages


def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names),
                 encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


readme = read("README.md")

install_requires = ["future",
                    "matplotlib",
                    "numpy",
                    "pytorch-ignite",
                    "torch>=1.1.0"]


setup(
    name='super_convergence',
    version='0.0.1',
    packages=find_packages(exclude=('tests', 'tests.*', 'data', 'examples')),
    url='https://github.com/ItamarWilf/pytorch-bonsai',
    license='MIT ',
    author='Itamar Wilf',
    author_email='',
    description='super converging pytorch-ignite trainers',
    long_description=readme,
    long_description_content_type="text/markdown",
    zip_safe=True,
    install_requires=install_requires
)
