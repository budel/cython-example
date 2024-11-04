import time
import unittest

import numpy as np
from naive_segmentation import runSequentialSegmentation
from segmentation_v1 import runSequentialSegmentation as runSegmentation1


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
        N = 1000
        img = np.random.randint(256, size=(N, N, 3), dtype=np.uint8)
        img = np.ascontiguousarray(img)
        means = np.random.randint(256, size=(self.cluster_n, 3)).astype(np.float64)
        means = np.ascontiguousarray(means)
        begin_time = time.time()
        segmentation0 = runSequentialSegmentation(img, means)
        print(f"Naive Segmentation took {time.time() - begin_time}")
        begin_time = time.time()
        segmentation1 = runSegmentation1(img, means)
        print(f"Segmentation1 took {time.time() - begin_time}")
        np.array_equal(segmentation0, segmentation1)


if __name__ == "__main__":
    unittest.main()
