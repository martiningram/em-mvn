from os import getenv
from setuptools import setup
from setuptools import find_packages


setup(
    name='em',
    version=getenv("VERSION", "LOCAL"),
    description='A number of EM algorithms for mixtures',
    packages=find_packages()
)
