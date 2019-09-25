from abc import ABCMeta, abstractmethod
import numpy as np

class NNLayer(metaclass=ABCMeta):
    """
    An abstract class that defines a neural network layer
    """
    @abstractmethod
    def forward_pass(self):
        """
        During forward pass the input is passed through all neurons
        in the layer

        :param input: The input is a matrix of shape (m x n) where n is the input dimension.
        m is the number of inputs in the set.
        :return: The output is a matrix of size (m x o) where o is the number of neurons in the
        layer
        """
        pass


class SimpleNNLayer(NNLayer):

    def __init__(self, activation_function, input_dimension, num_units):
        """
        CTOR

        :param activation_function: The activation function used for firing the neuron
        :param input_dimension: The dimension of input
        :param num_units: The number of neurons in the layer
        """
        self.__activation_function = activation_function
        self.__num_units = num_units
        self.__input_dimension = input_dimension
        self.__weight_matrix = np.random.rand(num_units, input_dimension)

    @property
    def num_units(self):
        return self.__num_units

    @property
    def input_dimension(self):
        return self.__input_dimension

    def forward_pass(self, input_matrix):
        assert(input.shape[1] == self.__num_units)
        activation_matrix = np.dot(input_matrix, self.__weight_matrix.T)
        return self.__activation_function(activation_matrix)

