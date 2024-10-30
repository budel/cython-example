import numpy as np


def runSequentialSegmentation(image, means):
    segmentation = [sequentialSegmentation(p, means) for p in image.reshape((-1, 3))]
    return segmentation

def sequentialSegmentation(p, means):
    mv = [np.linalg.norm(p - m) for m in means]
    return np.argmin(mv)
