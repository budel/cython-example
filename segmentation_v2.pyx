import numpy as np


def runSequentialSegmentation(image, means):
    tmp = image.reshape((-1, 3))
    cdef Py_ssize_t img_size = tmp.shape[0]
    segmentation = np.zeros(img_size, dtype=np.intc)
    for i in range(img_size):
        segmentation[i] = sequentialSegmentation(tmp[i], means)
    return segmentation

cdef sequentialSegmentation(p, means):
    cdef int i
    cdef Py_ssize_t means_size = means.shape[0]
    mv = np.zeros(means_size, dtype=np.intc)
    for i in range(means_size):
        mv[i] = np.linalg.norm(p - means[i])
    return np.argmin(mv)
