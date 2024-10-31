import time
import unittest

import numpy as np
from naive_segmentation import runSequentialSegmentation
from segmentation_v1 import runSequentialSegmentation as runSegmentation1
# from segmentation_v2 import runSequentialSegmentation as runSegmentation2
from segmentation_v2_without_numpy import runSequentialSegmentation as runSegmentation2
from segmentation_v3 import runSequentialSegmentation as runSegmentation3
from segmentation_v4 import runSequentialSegmentation as runSegmentation4
from segmentation_v5 import runSequentialSegmentation as runSegmentation5
 

class TestSegmentation(unittest.TestCase):
    def setUp(self):
        self.img = np.array(
            [[[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[8, 7, 6], [5, 4, 3], [2, 1, 0]]],
            dtype=np.uint8,
        )
        self.means = np.array([[1, 1, 1], [4, 4, 4], [7, 7, 7]], dtype=np.float64)
        self.cluster_n = 3

    def test_run(self):
        expected = [[0, 1, 2], [2, 1, 0]]
        segmentation = runSequentialSegmentation(self.img, self.means)
        np.array_equal(segmentation, expected)

    def test_regression(self):
        print("")
        N = 10**3
        img = np.random.randint(256, size=(N, N, 3), dtype=np.uint8)
        img = np.ascontiguousarray(img)
        means = np.random.rand(self.cluster_n, 3)
        means = np.ascontiguousarray(means)
        begin_time = time.time()
        segmentationSeq = runSequentialSegmentation(img, means)
        print(f"SeqSegmentation took {time.time() - begin_time}")
        begin_time = time.time()
        segmentation1 = runSegmentation1(img, means)
        print(f"Segmentation1 took {time.time() - begin_time}")
        np.array_equal(segmentationSeq, segmentation1)
        begin_time = time.time()
        segmentation2 = runSegmentation2(img, means)
        print(f"Segmentation2 took {time.time() - begin_time}")
        np.array_equal(segmentationSeq, segmentation2)
        begin_time = time.time()
        segmentation3 = runSegmentation3(img, means)
        print(f"Segmentation3 took {time.time() - begin_time}")
        np.array_equal(segmentationSeq, segmentation3)
        begin_time = time.time()
        segmentation4 = runSegmentation4(img, means)
        print(f"Segmentation4 took {time.time() - begin_time}")
        np.array_equal(segmentationSeq, segmentation4)
        begin_time = time.time()
        segmentation5 = runSegmentation5(img, means)
        print(f"Segmentation5 took {time.time() - begin_time}")
        np.array_equal(segmentationSeq, segmentation5)


if __name__ == "__main__":
    unittest.main()
