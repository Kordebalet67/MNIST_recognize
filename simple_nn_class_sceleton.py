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
        self.weights_hidden_output += self.learning_rate * np.dot((
            errors * last_output * (1 - last_output)),np.transpose(hidden_output)) 

        self.weights_input_hidden += self.learning_rate * np.dot(
            (hidden_errors * hidden_output * (1 - hidden_output)), np.transpose(inputs) 
                )

        return errors

    # here we'll ask our network for answers
    def query(self, inputs_list = None):
        inputs = np.random.rand(784,100) # temporary patch -> real data
        #inputs = np.array(inputs_list, ndmin=2).T
        print(inputs.shape)
        print("="*100)
        print("="*44 + ' Input data ' + "="*44)
        print(inputs)
        print("="*100)

        hidden_input = np.dot(self.weights_input_hidden, inputs)
        hidden_output = self.activation_function(hidden_input)

        last_input = np.dot(self.weights_hidden_output, hidden_output)
        last_output = self.activation_function(last_input)
        
        return last_output

def get_inputs_and_targets(path):
    input_list = []
    target_list = []
    with open(path, 'r') as file:
        inputs = file.readlines()
    inputs = inputs[:-1]
    for item in inputs:
        target_list.append(int(item[:1]))
        input_list.append(item[2:])

    for item in range(0, len(input_list)):
        input_list[item] = input_list[item].split(',')
        for i in range(0, len(input_list[item])):
            input_list[item][i] = int(input_list[item][i])

    

    return input_list, target_list





net = MNIST_Neural_Network(784, 100, 10)
# inputs, targets = get_inputs_and_targets('mnist_test_10.csv')
# print(inputs[0])
result = net.query()
print("="*100)
print("="*46 + ' Result ' + "="*46)
print(result)
print("="*100)

inputs, targets = get_inputs_and_targets('mnist_train_100.csv')

epochs = 50
error_value_list = []
for i in epochs:
    error_value_list.append(net.train(inputs[0], targets[0]))
min_error = min(error_value_list)
best_epoch_value = error_value_list.index(min_error) + 1

for i in range(best_epoch_value):
    net.train(inputs[0], targets[0])
net.query