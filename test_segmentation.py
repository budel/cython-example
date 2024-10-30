import time
import unittest

import numpy as np
from naive_segmentation import runSequentialSegmentation


class TestSegmentation(unittest.TestCase):
    def setUp(self):
        self.img = np.array(
            [[[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[8, 7, 6], [5, 4, 3], [2, 1, 0]]],
            dtype=np.uint8,
        )
        self.means = np.array([[1, 1, 1], [4, 4, 4], [7, 7, 7]], dtype=np.float64)
        self.mask = np.ones((self.img.shape[:2]), dtype=np.uint8)
        self.cluster_n = 3

    def test_run(self):
        expected = [[0, 1, 2], [2, 1, 0]]
        segmentation = runSequentialSegmentation(self.img, self.means)
        np.array_equal(segmentation, expected)

    def test_regression(self):
        print("")
        N = 10**6
        flat_img = np.random.randint(256, size=(N, 3), dtype=np.uint8)
        means = np.random.rand(self.cluster_n, 3)
        begin_time = time.time()
        segmentationSeq = runSequentialSegmentation(flat_img, means)
        print(f"SeqSegmentation took {time.time() - begin_time}")


if __name__ == "__main__":
    unittest.main()
