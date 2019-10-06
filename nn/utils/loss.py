"""
This package defines  different types of loss functions for neural network
"""

from abc import ABCMeta, abstractmethod

import numpy as np

class Loss(metaclass=ABCMeta):
    """
    An abstract class that defines the loss calculation
    and also the derivative of loss wrt to output
    """
    @abstractmethod
    def loss_value(self, expected_output, actual_output):
        """
        Subclasses should override this method to calculate the loss

        :param expected_output: The expected output
        :param actual_output: The actual output
        :return: The loss as calculated by the implementation of this function
        """
        pass

    @abstractmethod
    def gradient(self, expected_output, actual_output):
        """
        Returns the derivative of loss function wrt actual output

        :param expected_output:The expected output
        :param actual_output: The actual output
        :return: The gradient of loss function as calculated wrt actual output
        """
        pass

class SquaredLoss(Loss):
    """
    This class is an implementation of squared error loss function

    <code>loss = 1/2(expected - actual)^2</code>
    <code>derivative = actual - expected</code>
    """

    def loss_value(self, expected_output, actual_output):
        return np.square(expected_output - actual_output) / 2

    def gradient(self, expected_output, actual_output):
        return actual_output - expected_output
