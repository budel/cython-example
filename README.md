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
