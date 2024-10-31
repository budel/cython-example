import numpy as np
cimport cython

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def runSequentialSegmentation(unsigned char[:, :, :] image, double[:, :] means):
    cdef Py_ssize_t img_size0 = image.shape[0]
    cdef Py_ssize_t img_size1 = image.shape[1]
    segmentation = np.zeros((img_size0, img_size1), dtype=np.uint8)
    cdef unsigned char[:, :] seg_view = segmentation
    for i in range(img_size0):
        for j in range(img_size1):
            segmentation[i, j] = sequentialSegmentation(image[i, j], means)
    return segmentation

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef sequentialSegmentation(unsigned char[:] p, double[:, :] means):
    cdef unsigned char argmin 
    cdef double curmin = 2**8*2
    cdef int idx
    cdef int i
    cdef unsigned char p_i
    cdef double m_i
    cdef double sqnorm
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
