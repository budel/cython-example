from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Segmentation Example v6',
    ext_modules=cythonize("segmentation_v6.pyx"),
)

