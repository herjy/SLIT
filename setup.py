import os
import sys
from setuptools import find_packages

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))

setup(name='SLIT',
      version='0.1',
      description='Code for colour lens/source separation and lensed source reconstruction',
      author='Remy Joseph, Frederic Courbin, Jean-Luc Starck',
      author_email='remy.joseph@epfl.ch',
      # packages=['SLIT'],
      packages=find_packages(PACKAGE_PATH),
      zip_safe=False,
)
