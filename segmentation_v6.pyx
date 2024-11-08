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
    cdef Py_ssize_t img_size0 = image.shape[0]
    cdef Py_ssize_t img_size1 = image.shape[1]
    segmentation = np.zeros((img_size0, img_size1), dtype=np.uint8)
    segmentation = np.ascontiguousarray(segmentation)
    cdef unsigned char[:, ::1] seg_view = segmentation
    cdef unsigned char argmin
    cdef double curmin = 2**8*2
    cdef int idx
    cdef int k
    cdef unsigned char p_i
    cdef double m_i
    cdef double sqnorm
    cdef Py_ssize_t means_size = means.shape[0]
    cdef Py_ssize_t img_size2 = image.shape[2]

    for i in prange(img_size0, nogil=True, schedule='static'):
        for j in range(img_size1):
            for idx in range(means_size):
                sqnorm = 0
                for k in range(img_size2):
                    sqnorm = sqnorm + (image[i, j, k] - means[idx, i]) ** 2
                # norm = sqnorm ** 0.5
                if sqnorm < curmin:
                    argmin = idx
                    curmin = sqnorm
            seg_view[i, j] = argmin
    return segmentation
