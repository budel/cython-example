def runSequentialSegmentation(image, means):
    segmentation = [sequentialSegmentation(p, means) for p in image.reshape((-1, 3))]
    return segmentation


def sequentialSegmentation(p, means):
    argmin = -1
    curmin = 2**8 * 2
    for idx, m in enumerate(means):
        sqnorm = 0
        for p_i, m_i in zip(p, m):
            sqnorm += (p_i - m_i) ** 2
        # norm = sqnorm ** 0.5
        if sqnorm < curmin:
            argmin = idx
            curmin = sqnorm
    return argmin
