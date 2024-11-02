import time
import unittest

import numpy as np
from naive_segmentation import runSequentialSegmentation
from segmentation_v1 import runSequentialSegmentation as runSegmentation1
from segmentation_v2 import runSequentialSegmentation as runSegmentation2
from segmentation_v2_without_numpy import runSequentialSegmentation as runSegmentation2_without_numpy
from segmentation_v3 import runSequentialSegmentation as runSegmentation3
from segmentation_v4 import runSequentialSegmentation as runSegmentation4
from segmentation_v5 import runSequentialSegmentation as runSegmentation5
from segmentation_v6 import runSequentialSegmentation as runSegmentation6
from segmentation_v7 import runSequentialSegmentation as runSegmentation7
 

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
        begin_time = time.time()
        segmentation2 = runSegmentation2(img, means)
        print(f"Segmentation2 took {time.time() - begin_time}")
        np.array_equal(segmentation0, segmentation2)
        begin_time = time.time()
        segmentation2 = runSegmentation2_without_numpy(img, means)
        print(f"Segmentation2 without numpy took {time.time() - begin_time}")
        np.array_equal(segmentation0, segmentation2)
        begin_time = time.time()
        segmentation3 = runSegmentation3(img, means)
        print(f"Segmentation3 took {time.time() - begin_time}")
        np.array_equal(segmentation0, segmentation3)
        begin_time = time.time()
        segmentation4 = runSegmentation4(img, means)
        print(f"Segmentation4 took {time.time() - begin_time}")
        np.array_equal(segmentation0, segmentation4)
        begin_time = time.time()
        segmentation5 = runSegmentation5(img, means)
        print(f"Segmentation5 took {time.time() - begin_time}")
        np.array_equal(segmentation0, segmentation5)
#        begin_time = time.time()
#        segmentation6 = runSegmentation6(img, means)
#        print(f"Segmentation6 took {time.time() - begin_time}")
#        np.array_equal(segmentation0, segmentation6)
        begin_time = time.time()
        segmentation7 = runSegmentation7(img, means)
        print(f"Segmentation7 took {time.time() - begin_time}")
        np.array_equal(segmentation0, segmentation7)


if __name__ == "__main__":
    unittest.main()
