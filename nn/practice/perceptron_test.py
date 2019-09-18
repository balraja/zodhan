# This file contains an example for testing perceptron
# based on the sonar data as mentioned in
# https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)

from zodhan.nn.data.sonar import SonarData
from zodhan.nn.perceptron import Perceptron, PerceptronTrainer

def perceptron_test():
    sonar_data = SonarData()
    sonar_data.load_data()
    trainer = PerceptronTrainer(
         sonar_data.get_input_data(0,1,2), sonar_data.get_output_vector(), 0.015, 1000)
    perceptron = trainer.train()

    test_data = sonar_data.get_input_data(0, 1, 2)
    output_vector = sonar_data.get_output_vector()
    for index in range(test_data.shape[0]):
        actual_type = sonar_data.classified_object(perceptron.evaluate(test_data[index]))
        expected_type = sonar_data.classified_object(output_vector[index])
        print(f"Expected {expected_type} Actual {actual_type}")

if __name__ == "__main__":
    perceptron_test()