import timeit


def test_regression():
    print("")
    s = """
import numpy as np
from naive_segmentation import runSequentialSegmentation
from segmentation_v1 import runSequentialSegmentation as runSegmentation1
# from segmentation_v2 import runSequentialSegmentation as runSegmentation2
from segmentation_v2_without_numpy import runSequentialSegmentation as runSegmentation2
from segmentation_v3 import runSequentialSegmentation as runSegmentation3
from segmentation_v4 import runSequentialSegmentation as runSegmentation4
from segmentation_v5 import runSequentialSegmentation as runSegmentation5
from segmentation_v6 import runSequentialSegmentation as runSegmentation6
from segmentation_v7 import runSequentialSegmentation as runSegmentation7

cluster_n = 3
N = 10000
img = np.random.randint(256, size=(N, N, 3), dtype=np.uint8)
img = np.ascontiguousarray(img)
means = np.random.randint(256, size=(cluster_n, 3)).astype(np.float64)
means = np.ascontiguousarray(means)
"""
    #    it, t = timeit.Timer('runSequentialSegmentation(img, means)', setup=s).autorange()
    #    print(f"Naive per run: {t/it}")
    #    it, t = timeit.Timer('runSegmentation1(img, means)', setup=s).autorange()
    #    print(f"v1 per run: {t/it}")
    #    it, t = timeit.Timer('runSegmentation2(img, means)', setup=s).autorange()
    #    print(f"v2 per run: {t/it}")
    it, t = timeit.Timer("runSegmentation3(img, means)", setup=s).autorange()
    print(f"v3 per run: {t/it}")
    it, t = timeit.Timer("runSegmentation4(img, means)", setup=s).autorange()
    print(f"v4 per run: {t/it}")
    #    it, t = timeit.Timer('runSegmentation5(img, means)', setup=s).autorange()
    #    print(f"v5 per run: {t/it}")
    #    it, t = timeit.Timer('runSegmentation6(img, means)', setup=s).autorange()
    #    print(f"v6 per run: {t/it}")
    it, t = timeit.Timer("runSegmentation7(img, means)", setup=s).autorange()
    print(f"v7 per run: {t/it}")
