from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Segmentation Example v2',
    ext_modules=cythonize("segmentation_v2_without_numpy.py"),
)

