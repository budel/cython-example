# cython: infer_types=True
import numpy as np
cimport cython

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def runSequentialSegmentation(unsigned char[:, :, ::1] image, double[:, ::1] means):
    img_size0 = image.shape[0]
    img_size1 = image.shape[1]
    segmentation = np.zeros((img_size0, img_size1), dtype=np.uint8)
    segmentation = np.ascontiguousarray(segmentation)
    cdef unsigned char[:, ::1] seg_view = segmentation
    for i in range(img_size0):
        for j in range(img_size1):
            seg_view[i, j] = sequentialSegmentation(image[i, j], means)
    return segmentation

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef sequentialSegmentation(unsigned char[::1] p, double[:, ::1] means):
    curmin = 2.0**8*2
    means_size = means.shape[0]
    p_size = p.shape[0]

    for idx in range(means_size):
        sqnorm = 0
        for i in range(p_size):
            sqnorm += (p[i] - means[idx, i]) ** 2
        # norm = sqnorm ** 0.5
        if sqnorm < curmin:
            argmin = idx
            curmin = sqnorm
    return argmin
