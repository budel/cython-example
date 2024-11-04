from setuptools import setup
from Cython.Build import cythonize

setup(
    name="Segmentation Examples",
    ext_modules=cythonize(
        ["*.pyx"],
        annotate=True,
    ),
)
