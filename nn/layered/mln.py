from zodhan.nn.layered.layer import NNLayer, SimpleNNLayer

class MultiLayerNetwork(object):
    """
    This class defines a multi layered neural network
    """

    def __init__(self):
        """
        CTOR
        """
        self.__layers = []

    def add_layer(self, layer):
        """
        Adds a neural network layer for processing the input data

        :param layer: The layer of nn for processing the data.
        :return: None
        """

        if len(self.__layers) > 0:
            last_layer = self.__layers[-1]
            assert(last_layer.num_units == layer.input_dimension)
        self.__layers.append(layer)

    def forward_pass(self, input_matrix):
        """
        During the forward pass the data will be processed
        through various nn layers and final output after
        processing through various layers would be returned

        :param input: Matrix of shape (m x n) where m is the number of samples and n is the dimension of input
        :return: output as a matrix of shape (m x o) where m is number of samples and o is the dimnesion of the output layer
        """
        output_matrix = input_matrix
        for layer in self.__layers:
            output_matrix = layer.forward_pass(output_matrix)
        return output_matrix

class MLNBuilder(object):
    """
    This class defines a builder for multi layer network
    """

    def __init__(self):
        self.__mln = MultiLayerNetwork()

    def add_layer(self, input_dimension, num_units, activation_function):
        self.__mln.add_layer(SimpleNNLayer(input_dimension, num_units, activation_function))

    def build(self):
        return self.__mln