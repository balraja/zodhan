from zodhan.nn.layered.layer import NNLayer, SimpleNNLayer

import numpy as np
import sys

class MultiLayerNetwork(object):
    """
    This class defines a multi layered neural network
    """

    def __init__(self):
        """
        CTOR
        """
        self.__layers = []

    @property
    def layers(self):
        return self.__layers

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
        activations = []
        activation = input_matrix
        for layer in self.__layers:
            activations.append(activation)
            activation = layer.forward_pass(activation)
        return (activation, activations)

    def backward_pass(self, output_loss, layer_inputs):

        reversed_layers = self.__layers[::-1]
        reversed_inputs = layer_inputs[::-1]
        derivative = output_loss
        for i in range(len(reversed_layers)):
            layer = reversed_layers[i]
            layer_input = reversed_inputs[i]
            derivative = layer.backward_pass(derivative, layer_input)

    def __str__(self):
        return "\n".join([str(x) for x in self.__layers])

class MLNBuilder(object):
    """
    This class defines a builder for multi layer network
    """

    def __init__(self, learning_rate):
        self.__mln = MultiLayerNetwork()
        self.__learning_rate = learning_rate

    def add_layer(self, input_dimension, num_units, activation_function):
        self.__mln.add_layer(SimpleNNLayer(activation_function, input_dimension, num_units, self.__learning_rate))

    def build(self):
        return self.__mln

class MLNTrainer(object):

    def __init__(self, mln, loss_function, input, output, batch_size=1000, max_epocs=100, stopping_threshold=0.01):
        """
        CTOR

        :param mln: The multi layered network
        :param loss_function : The loss function to be used for measuring the loss
        :param input: The input matrix of dimansion m x n where m is number
                      of test data and n is the dimension of matrix
        :param output : The expected output for training input
        :param batch_size : The batch size to be used for stochastic gradient descent
        :param max_epocs : The maximum number of epochs
        :param stopping_threshold : The stopping threshold to be used for stopping
                                    Its applied like l2norm(loss_vector) < threshold
        """
        self.__mln = mln
        self.__loss_function = loss_function
        self.__input = input
        self.__output = output
        self.__batch_size = batch_size
        self.__max_epochs = max_epocs
        self.__stopping_threshold = stopping_threshold

    def train(self):
        """
        This function trains the multi layer perceptron for the
        provided input by chunking the input into mini batches
        and performs both forward and backward propagation in
        MLN

        :return: The trained mln
        """
        assert(self.__input.shape[1] == self.__mln.layers[0].input_dimension)
        num_batches = self.__input.shape[0] // self.__batch_size
        input_batches = np.vsplit(self.__input, num_batches)
        output_batches = np.vsplit(self.__output, num_batches)

        num_epoch = 0
        l2_loss = sys.float_info.max
        while num_epoch < self.__max_epochs:
            print(f"Starting Epoch {num_epoch}")
            batch_loss_norms = []
            for i in range(num_batches):
                input = input_batches[i]
                expected_output = output_batches[i]
                #forward_pass
                (actual_output, layer_inputs) = self.__mln.forward_pass(input)
                print(actual_output)
                loss_vector = self.__loss_function.loss_value(expected_output, actual_output)
                print(f"The loss of batch {i} in epoch {num_epoch} is {loss_vector}")
                # backward_pass
                loss_derivative = self.__loss_function.gradient(expected_output, actual_output)
                self.__mln.backward_pass(loss_derivative, layer_inputs)
                batch_loss_norms.append(np.linalg.norm(loss_vector))
            l2_loss = np.linalg.norm(batch_loss_norms)
            if l2_loss < self.__stopping_threshold:
                break
            num_epoch += 1

        return self.__mln
