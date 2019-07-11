import unittest
import similarity.cos as cos
import numpy as np


class TestCos(unittest.TestCase):
    def setUp(self) -> None:
        self.mat = np.array([1, 2, 3])
        self.lst = np.array([[1, 2, 2.5], [1, 2, 3], [1, 3, 2]])

    def test_max_sim_succeed(self):
        idx, cos_max = cos.max_sim(self.mat, self.lst)
        self.assertEqual(idx, 1)
        self.assertTrue((self.mat == self.lst[idx]).all())


if __name__ == "__main__":
    unittest.main()


