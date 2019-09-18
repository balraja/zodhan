"""
This class defines the base classes of a neural network
that's used for performing deep learning
"""

from abc import  ABCMeta, abstractmethod
class BaseNN(metaclass=ABCMeta):
    """
    The base class of a neural network
    """

    @abstractmethod
    def train(self, input_array, expected_output, epoch):
        """
        This method is used for training a neural network.

        :param input_array: The training input as a ndarray
        :param expected_output: The expected output as scalar or nd array
        :return: Nil
        """
        pass

    @abstractmethod
    def evaluate(self, input_array):
        """
        This method is used for evaluating the input post training

        :param input_array: The input as ndarray
        :return: The expected output as ndarray or scalar
        """
        pass

class Epoch(object):
    """
    This class defines a epoch of a training. This is used
    for tracking the number of misclassified inputs during training
    """

    def __init__(self, training_set_size):
        """
        CTOR
        :param training_set_size: The size of the training set.
        :param epoch_number: The epoch number or the iteration number
        """
        self.__training_set_size = training_set_size
        self.__epoch_number = 0
        self.__misclassified_input_count = 0.0

    @property
    def epoch_number(self):
        return self.__epoch_number

    def new_epoch(self):
        self.__epoch_number += 1
        self.__misclassified_input_count = 0.0

    def update_error_count(self, error_Count = 1.0):
        self.__misclassified_input_count += error_Count

    def error_count(self):
        return self.__misclassified_input_count

    def error_rate(self):
        return self.__misclassified_input_count / self.__training_set_size




