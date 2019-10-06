from abc import ABCMeta, abstractmethod
import numpy as np

class NNLayer(metaclass=ABCMeta):
    """
    An abstract class that defines a neural network layer
    """
    @abstractmethod
    def forward_pass(self, input_matrix):
        """
        During forward pass the input is passed through all neurons
        in the layer

        :param input_matrix: The input is a matrix of shape (m x n) where n is the input dimension.
        m is the number of inputs in the set.
        :return: The output is a matrix of size (m x o) where o is the number of neurons in the
        layer
        """
        pass

    def backward_pass(self, derivative_matrix, input_matrix):
        """
        During backward propagation, we propagate the derivative back through
        this layer

        :param derivative:  A matrix (m x o) of derivatives where m is the input
                            batch size and o is number of neurons in this layer
        :param input_matrix: The input is a matrix of shape (m x n) where n is
                             the input dimension and m is the number of inputs in
                             the set.
        :return: Derivatives matrix of shape (m x n) where m is the batch size and
                 n is number of inputs for this layer
        """
        pass


class SimpleNNLayer(NNLayer):

    def __init__(self, activation_function, input_dimension, num_units, learning_rate):
        """
        CTOR

        weight_matrix is of shape (o = num_units x n = input_dimension)

        :param activation_function: The activation function used for firing the neuron
        :param input_dimension: The dimension of input
        :param num_units: The number of neurons in the layer
        :param learning_rate : The rate at which weight update happens
        """
        self.__activation_function = activation_function
        self.__num_units = num_units
        self.__input_dimension = input_dimension
        self.__weight_matrix = np.random.rand(num_units, input_dimension)
        self.__learning_rate = learning_rate
        self.__cached_z = None

    def __str__(self):
        return f"{self.__weight_matrix}"

    @property
    def num_units(self):
        return self.__num_units

    @property
    def input_dimension(self):
        return self.__input_dimension

    def forward_pass(self, input_matrix):
        assert(input_matrix.shape[1] == self.__input_dimension)
        assert(self.__cached_z is None)
        z_matrix = np.dot(input_matrix, self.__weight_matrix.T)
        a_matrix = self.__activation_function(z_matrix)
        self.__cached_z = z_matrix
        return a_matrix

    def backward_pass(self, derivative_matrix, input_matrix):
        assert(self.__cached_z is not None)
        activation_gradient_matrix = self.__activation_function.gradient(self.__cached_z)
        # Hadamard product of derivatives with activation_gradient
        # The matrix is of dimension (m = batch_size x o = num_units)
        z_derivative_matrix = np.multiply(activation_gradient_matrix, derivative_matrix)
        # The weight derivatives are of size o x n equivalent to
        # (num_units_in_layer x input_dimension)
        w_derivative_matrix = np.dot(z_derivative_matrix.T, input_matrix)
        self.__weight_matrix = self.__weight_matrix - self.__learning_rate * w_derivative_matrix
        # The input derivative matrix is the delta for each of inputs
        # fed from the previous layer. Its of dimension = (batch_size x num_inputs)
        # This is acheived by multiplying w.T = (num_inputs X num_units)
        # with z_derivative.T = (num_units x batch_size) and taking transpose of the
        # result
        input_derivative_matrix = np.dot(self.__weight_matrix.T, z_derivative_matrix.T).T
        self.__cached_z = None
        return input_derivative_matrix

