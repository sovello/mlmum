import numpy as np

import os


class SequentialPerceptron:
    """ A basic Perceptron """
    def __init__(self, inputs, targets):
        """ Constructor """
        # set the network size
        if np.ndim(inputs) > 1:
            self.number_of_inputs = np.shape(inputs)[1]
            
        else:
            self.number_of_inputs = 1

        if np.ndim(targets) > 1:
            self.number_of_outputs = np.shape(targets)[1]
        else:
            self.number_of_outputs = 1

        self.data_dimensions = np.shape(inputs)[0]

        # initialize the network
        self.weights = np.random.rand(self.number_of_inputs+1, \
                                          self.number_of_outputs) * 0.1 - 0.5
        #print self.weights
        
    def train_perceptron(self, inputs, targets, learning_rate, iterations):
        """ Training function 
        Loops throught the number of iterations for the given inputs and the targets
        """
        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs, np.ones((self.data_dimensions, 1))), axis = 1)
        for i in range(iterations):
            # sequential implementation the perceptron training
            for r in range(np.shape(inputs)[0]): # go through every row
                row = inputs[r] # get rows
                for i in range(np.shape(row)[0]): # go through each element in row
                    wt = self.weights[i][0]
                    activation = row[i] * wt # activation
                    output = 1 if activation > 0 else  0 # output
                    self.weights[i][0] = wt - learning_rate * (output - targets[r])*row[i] # calculate new weights

       
    def perceptron_predict(self, inputs):
        """ Run the network forward.
        This function is used to predict for new values after the training has been done
        """
        # Compute activations

        activations = np.dot(inputs, self.weights)

        #Determine/predict the final output for this input,
        return np.where(activations > 0, 1, 0)

    def confusion_matrix(self, inputs, targets):
        # Add the inputs that match the bias node
        inputs = np.concatenate(( inputs, \
                                      -np.ones((inputs.shape[0], 1))), axis = 1)

        outputs = np.dot(inputs, self.weights)

        nClasses = np.shape(targets)[1]
        
        if nClasses == 1:
            nClasses = 2
            outputs = np.where(outputs > 0, 1, 0)

        else:
            # 1-of-N necoding
            outputs = np.argmax(outputs, 1)
            targets = np.argmax(targets, 1)

        cm = np.zeros((nClasses, nClasses))
        for i in range(nClasses):
            for j in range(nClasses):
                cm[i, j] = np.sum(np.where(outputs == i, 1, 0) *\
                                      np.where(targets == j, 1, 0))
        return cm