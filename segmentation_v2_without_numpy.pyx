import numpy as np

def runSequentialSegmentation(image, means):
    tmp = image.reshape((-1, 3))
    cdef Py_ssize_t img_size = tmp.shape[0]
    segmentation = np.zeros(img_size, dtype=np.intc)
    for i in range(img_size):
        segmentation[i] = sequentialSegmentation(tmp[i], means)
    return segmentation

cdef sequentialSegmentation(p, means):
    cdef int argmin = -1
    cdef int curmin = 2**8*2
    cdef int idx
    cdef int p_i
    cdef int m_i
    cdef int sqnorm
    cdef Py_ssize_t means_size = means.shape[0]

    for idx in range(means_size):
        for p_i, m_i in zip(p, means[idx]):
            sqnorm = p_i * m_i
        # norm = sqnorm ** 0.5
        if sqnorm < curmin:
            argmin = idx
            curmin = sqnorm
    return argmin
