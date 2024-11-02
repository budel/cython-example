from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Segmentation Examples',
    ext_modules=cythonize([
        "segmentation_v2.pyx",
        "segmentation_v2_without_numpy.pyx",
        "segmentation_v3.pyx",
        "segmentation_v4.pyx",
        "segmentation_v5.pyx",
        "segmentation_v6.pyx",
    ],
    annotate=True,
    )
)

