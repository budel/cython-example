# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp

import numpy as np
cimport cython
from cython.parallel import prange


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def runSequentialSegmentation(unsigned char[:, :, ::1] image, double[:, ::1] means):
    cdef int i
    cdef int j
    cdef unsigned char retval
    cdef Py_ssize_t img_size0 = image.shape[0]
    cdef Py_ssize_t img_size1 = image.shape[1]
    segmentation = np.zeros((img_size0, img_size1), dtype=np.uint8)
    segmentation = np.ascontiguousarray(segmentation)
    cdef unsigned char[:, ::1] seg_view = segmentation
    cdef unsigned char[:, :, ::1] img_view = image
    for i in prange(img_size0, nogil=True):
        for j in range(img_size1):
            #seg_view[i, j] = sequentialSegmentation(img_view[i, j], means)
            retval = sequentialSegmentation(img_view[i, j], means)
    return segmentation

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef unsigned char sequentialSegmentation(unsigned char[::1] p, double[:, ::1] means) noexcept nogil:
    cdef unsigned char argmin
    cdef double curmin = 2**8*2
    cdef int idx
    cdef int i
    cdef unsigned char p_i
    cdef double m_i
    cdef double sqnorm = 0
    cdef Py_ssize_t means_size = means.shape[0]
    cdef Py_ssize_t p_size = p.shape[0]

    for idx in range(means_size):
        for i in range(p_size):
            sqnorm = p[i] * means[idx, i]
        # norm = sqnorm ** 0.5
        if sqnorm < curmin:
            argmin = idx
            curmin = sqnorm
    return argmin
