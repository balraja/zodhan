# This module contains all code for implementing a perceptron

import numpy as np

from zodhan.nn.base_nn import BaseNN, Epoch
from zodhan.nn.utils.activation import SigmoidFunction

class Perceptron(BaseNN):
    """
    Defines a perceptron. A perceptron is a single layer network
    that can be used for learning linearly separable classification
    """
    def __init__(self, num_inputs, activation_function, learning_rate=0.001):
        self.__num_inputs = num_inputs
        self.__activation_function = activation_function
        self.__weight_vector = np.random.random(num_inputs)
        self.__updated_weight_vector = self.__weight_vector.copy()
        self.__learning_rate = learning_rate

    @property
    def num_inputs(self):
        return self.__num_inputs

    @property
    def weight_vector(self):
        return self.__weight_vector

    def train(self, input_array, expected_output, epoch):
        """
        Trains the perceptron as follows

        if activation(weight * input) != expected_output
            weight += activation(weight * input) * expected_output * input
        """
        activation_output = self.__activation_function(np.dot(input_array, self.__weight_vector))
        output = Perceptron.bound_to_1_and_minus_1(activation_output)
        #print(f"epoch {epoch.epoch_number} expected {expected_output} actual {output} {input_array} {self.__weight_vector} {self.__updated_weight_vector} {activation_output}")
        if output != expected_output:
            update = self.__learning_rate * input_array
            if expected_output > 0:
                self.__updated_weight_vector += update
            else:
                self.__updated_weight_vector -= update
            #print(f"weight {self.__weight_vector} updated {self.__updated_weight_vector}")
            epoch.update_error_count()

    def update_weight_vector(self):
        print(f"update weight to {self.__updated_weight_vector}")
        self.__weight_vector = self.__updated_weight_vector.copy()

    def evaluate(self, input_array):
        output = self.__activation_function(np.dot(input_array, self.__weight_vector))
        print(f"{self.__weight_vector} {input_array} {output}")
        return Perceptron.bound_to_1_and_minus_1(output)

    @staticmethod
    def bound_to_1_and_minus_1(value):
        if value >= 0.5:
            return 1
        else:
            return -1


class PerceptronTrainer(object):
    """
    Defines the trainer for a perceptron. A perceptron trainer
    accepts expected input and output as numpy arrays
    and trains a perceptron for each input row. The stopping
    criterion is based on error rate on an epoch. if error rate
    on epoch is less than certain threhold training is stopped.
    """

    def __init__(self, input_matrix, output_vector, stopping_threshold=0.01, max_iterations = 1000):
        assert input_matrix.ndim == 2
        self.__input_matrix = input_matrix
        assert output_vector.ndim == 1
        self.__output_vector = output_vector
        self.__stopping_threshold = stopping_threshold
        self.__max_iterations = max_iterations
        self.__perceptron = Perceptron(input_matrix.shape[1], SigmoidFunction())

    def train(self):
        epoch = Epoch(self.__output_vector.shape[0])
        while True:
            for index in range(self.__output_vector.shape[0]):
                self.__perceptron.train(
                    self.__input_matrix[index], self.__output_vector[index], epoch)
            print(f"epoch {epoch.epoch_number} error cnt {epoch.error_count()} rate {epoch.error_rate()}")
            if epoch.epoch_number >= self.__max_iterations or epoch.error_rate() < self.__stopping_threshold:
                break
            else:
                epoch.new_epoch()
                self.__perceptron.update_weight_vector()
        print(f"Perceptron weight {self.__perceptron.weight_vector}")
        return self.__perceptron









