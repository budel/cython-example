# Cython Example
Example application to show the usage of cython.

## Setup
```
python -m venv --system-site-packages ex
. ./ex/bin/activate
pip install -r requirements.txt
```

## First test
```
. ./ex/bin/activate
pytest -s
```

## Compile with cython
```
. ./ex/bin/activate
cython naive_segmentation.py
```
generates a c file, which needs to be compiled to a dynamic library.

```
. ./ex/bin/activate
cythonize naive_segmentation.py
```
generates a c file, and compiles it to a dynamic library.
The flag -a also generates an annotated html file.

Writing a `setup.py` with `setuptools` is the recommended approach. See the example `setup.py` for more infos and run
```
python setup.py build_ext --inplace
```
to generate a c file and a compiled library inside the source folder.
It has the advantage of compiling multiple files with one command.

## Optimization
Let's get rid of the numpy dependency. A look at the [source](https://github.com/numpy/numpy/blob/v2.1.0/numpy/linalg/_linalg.py#L2624-L2883) reveals that the 2-norm is used.
A numpy free version is implemented in `segmentation_v1.py` Notice how we could skip the costly square function, as we do not need it for comparison.
