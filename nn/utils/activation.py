"""
This packages defines the different types of activation functions
of a neural network
"""
import math
import numpy as np

from abc import ABCMeta, abstractmethod
from enum import Enum

class ActivationFunctionEnum(Enum):
    """
    An enum that enumerates different types of activation
    functions.
    """
    SIGMOID = 1
    TANH = 2
    RELU = 3

class ActivationFunction(metaclass=ABCMeta):
    """
    Defines a blue print for an utils function
    of a neuron
    """

    @abstractmethod
    def activate(self, input):
        """
        subclasses should override this method to defines the
        logic of utils function
        :return: a double specifying 
        """
        pass

    def __call__(self, input):
        return self.activate(input)



class SigmoidFunction(ActivationFunction):
    """
    Implements an ActivationFunction as a sigmoid function
    to provide a smooth and continous transition from 0 to 1.
    """

    def activate(self, input):
        if isinstance(input, np.ndarray):
            return SigmoidFunction.sigmoid_vector(input)
        else:
            return SigmoidFunction.sigmoid_scalar(input)

    @staticmethod
    def sigmoid_scalar(x):
        """
        This function calculates the sigmoid for every value in vector x.
        The sigmoid as per https://en.wikipedia.org/wiki/Sigmoid_function
        is  1 / 1 + e(-x). It can also be represented as e(x)/e(x) + 1

        :param x: The input scalar
        :return:  sigmoid of input.
        """
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def sigmoid_vector(x):
        """
        This function calculates the sigmoid for every value in vector x.
        The sigmoid as per https://en.wikipedia.org/wiki/Sigmoid_function
        is  1 / 1 + e(-x). It can also be represented as e(x)/e(x) + 1

        :param x: The input vector
        :return: vector wherein each value represents the sigmoid of its value.
        """
        exp_vector = np.exp(x)
        exp_vector_plus_1 = 1 + exp_vector
        return exp_vector / exp_vector_plus_1

class UnknownActivationFunctionEnumError(Exception):
    """
    An exception class to denote error wrt activation functions
    """
    pass


def get_activation_function(activation_function_enum):
    if activation_function_enum == ActivationFunctionEnum.SIGMOID:
        return SigmoidFunction()
    raise UnknownActivationFunctionEnumError(f"Unknown activation function {activation_function_enum}")
