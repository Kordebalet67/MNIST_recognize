import numpy as np
import math as m
import scipy.special
import joblib


class MNIST_Neural_Network:
    """
    Base class for our own neural network.
    Contains from three layers with only one hidden.
    """

    # constructor method, arguments:
    # input_nodes, hidden_nodes, output_nodes, learning_rate
    def __init__(self, input_nodes = 3, hidden_nodes = 3, 
                 output_nodes = 3, learning_rate = 0.5,
                   epochs = 5):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights_input_hidden = np.random.normal(0.0, pow(self.input_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.weights_hidden_output = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

        self.activation_function = lambda x: scipy.special.expit(x)

    # here we'll make all settings and train features of our network
    def train(self, training_list_path = "mnist_train_100.csv"):
        for e in range(self.epochs):
            print(f"Epoch {e} / {self.epochs}", end='\r')
            with open(training_list_path, 'r') as file:
                training_list = file.readlines()

            # go through all records in the training data set
            for record in training_list:
                print(f"{training_list.index(record)} / {len(training_list)}", end='\r')
                # split the record by the ',' commas
                all_values = record.split(',')
                # scale and shift the inputs
                inputs = (np.asarray(all_values[1:], dtype=float) / 255.0 * 0.99) + 0.01
                # create the target output values (all 0.01, except the desired label which is 0.99)
                targets = np.zeros(10) + 0.01
                # all_values[0] is the target label for this record
                targets[int(all_values[0])] = 0.99
                
                # convert inputs list to 2d array -> WE DON'T NEED TO GET MATRIX. WE NEED VERTICAL VECTOR WITH LENGTH = NUMBER OF INPUT_NODES
                inputs = np.array(inputs, ndmin=2).T
                targets = np.array(targets, ndmin=2).T
                
                # calculate signals on hidden layer - inputs and outputs
                hidden_inputs = np.dot(self.weights_input_hidden, inputs)
                hidden_outputs = self.activation_function(hidden_inputs)
                
                # calculate signals on final output layer - inputs and outputs
                final_inputs = np.dot(self.weights_hidden_output, hidden_outputs)
                final_outputs = self.activation_function(final_inputs)
                
                
                output_errors = targets - final_outputs
                hidden_errors = np.dot(self.weights_hidden_output.T, output_errors) 
                
                # update the weights for the links between the hidden and output layers
                self.weights_hidden_output += self.learning_rate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
                self.weights_input_hidden += self.learning_rate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
                
        return self.weights_input_hidden, self.weights_hidden_output

    # here we'll ask our network for answers
    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_input = np.dot(self.weights_input_hidden, inputs)
        hidden_output = self.activation_function(hidden_input)

        last_input = np.dot(self.weights_hidden_output, hidden_output)
        last_output = self.activation_function(last_input)
        
        return last_output


    def score_info(self, test_data_list):
        # scorecard for how well the network performs, initially empty
        scorecard = []

        # go through all the records in the test data set
        for record in test_data_list:
            # split the record by the ',' commas
            all_values = record.split(',')
            # correct answer is first value
            correct_label = int(all_values[0])
            # scale and shift the inputs
            inputs = (np.asarray(all_values[1:],  dtype=float) / 255.0 * 0.99) + 0.01
            # query the network
            outputs = self.query(inputs)
            # the index of the highest value corresponds to the label
            label = np.argmax(outputs)
            # append correct or incorrect to list
            if (label == correct_label):
                # network's answer matches correct answer, add 1 to scorecard
                scorecard.append(1)
            else:
                # network's answer doesn't match correct answer, add 0 to scorecard
                scorecard.append(0)
        scorecard_array = np.asarray(scorecard)
        final_score = scorecard_array.sum() / scorecard_array.size
        print (f"performance = {final_score}" )
        pass

    def save_weights(self, file_name: str):
        w_dict = {'ih': self.weights_input_hidden,
        'ho': self.weights_hidden_output}
        joblib.dump(w_dict, file_name)
        pass

    def load_weights(self, file_name: str):
        load_weights = joblib.load(file_name)
        self.weights_input_hidden = load_weights['ih']    
        self.weights_hidden_output = load_weights['ho']
        pass

    def return_answers(self, input_data):
        answers = self.query(input_data)
        answer_list = list(answers)
        return answer_list.index(max(answer_list)), (np.float64(max(answer_list)) * 100)

    # current state is only for JPG/JPEG format with 28x28 px size
    def input_from_picture(self, picture_path):
        from PIL import Image, ImageOps # pip install pillow
        img = Image.open(picture_path).convert("L")
        inverted_image = ImageOps.invert(img)
        pixels = list(inverted_image.get_flattened_data())
    
        pixels = (np.asarray(pixels,  dtype=float) / 255.0 * 0.99) + 0.01
        return self.return_answers(pixels)

