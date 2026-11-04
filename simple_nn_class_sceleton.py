import numpy as np
import math as m

class MNIST_Neural_Network:
    """
    base class for our own neural network.
    """

    # constructor method, arguments:
    # input_nodes, hidden_nodes, output_nodes, learning_rate
    def __init__(self, input_nodes = 3, hidden_nodes = 3, 
                 output_nodes = 3, learning_rate = 0.5):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.learning_rate = learning_rate

        self.weights_input_hidden = np.random.rand(self.input_nodes, self.hidden_nodes) - 0.5
        self.weights_hidden_output = np.random.rand(self.hidden_nodes, self.output_nodes) - 0.5

        self.activation_function = lambda x: 1 / (1+np.exp(-x))

    # here we'll make all settings and train fatures of our network
    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_input = np.dot(self.weights_input_hidden, inputs)
        hidden_output = self.activation_function(hidden_input)

        last_input = np.dot(self.weights_hidden_output, hidden_output)
        last_output = self.activation_function(last_input)

        errors = targets - last_output
        print("="*100)
        print(f'Mistake: {errors}')
        print("="*100)

        hidden_errors = np.dot(self.weights_hidden_output.T, errors)
        self.weights_hidden_output += self.learning_rate * np.dot((errors * last_output * (1 - last_output)),
                                                             np.transpose(hidden_output)) 
        # input_errors = np.dot()

    # here we'll ask our network for answers
    def query(self, inputs = None):
        inputs = np.random.rand(3,3) # temporary patch -> real data
        print("="*100)
        print("="*44 + ' Input data ' + "="*44)
        print(inputs)
        print("="*100)

        hidden_input = np.dot(self.weights_input_hidden, inputs)
        hidden_output = self.activation_function(hidden_input)

        last_input = np.dot(self.weights_hidden_output, hidden_output)
        last_output = self.activation_function(last_input)

        return last_output


net = MNIST_Neural_Network()
result = net.query()
print("="*100)
print("="*46 + ' Result ' + "="*46)
print(result)
print("="*100)