import numpy as np
import unittest

from math import exp
from zodhan.nn.utils.sigmoid import sigmoid_scalar, sigmoid_vector

class SigmoidTest(unittest.TestCase):

    def test_sigmoid_scalar(self):
        expected_output = 1 / (1 + exp(-1))
        self.assertEquals(expected_output, sigmoid_scalar(1))

    def test_sigmoid_vector(self):
        input_array = [1.0, 2.0]
        expected_output_array = np.round(np.array([1 / (1 + exp(-x))
                                          for x in input_array],
                                         dtype=np.float), 5)
        actual_output_array = \
            np.round(sigmoid_vector(np.array(input_array, dtype=np.float)), 5)
        self.assertTrue(np.array_equal(
            expected_output_array, actual_output_array))


if __name__ == '__main__':
    unittest.main()
