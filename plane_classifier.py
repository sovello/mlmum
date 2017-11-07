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
        
if __name__ == "__main__":

    path = os.getcwd()
    # encode Bomber:0, Fighter:1
    data = np.loadtxt(path+'/plane_class_data.csv',
                          skiprows=1,
                          delimiter=",",
                          converters =
                          {2: lambda plane:

                          1 if plane == "Bomber" else 0})

    print "All data"
    print data
    training_data = data[:5,:2]
    print "Training data"
    training_target = data[:5, 2:3]
    print training_data
    print "Test data"
    test_data = data[5:, :2]
    print test_data
    test_target = data[5:, 2:3]
    print test_target
    # train
    
    perceptron = SequentialPerceptron(training_data, training_target)

    # predict
    perceptron.train_perceptron(training_data, training_target, 0.25, 3)

    # print matrix
    print perceptron.confusion_matrix(test_data, test_target)

    
    training_data = data[[0, 1, 2,  4, 8, 9, 6, 3],:2]
    print "Training data"
    training_target = data[[0, 1, 2,  4, 8, 9, 6, 3], 2:3]
    print training_data
    print "Test data"
    test_data = data[[5, 7], :2]
    print test_data
    test_target = data[[5, 7], 2:3]
    print test_target
    # train
    
    perceptro = SequentialPerceptron(training_data, training_target)

    # predict
    perceptro.train_perceptron(training_data, training_target, 0.25, 10)

    # print matrix
    print perceptro.confusion_matrix(test_data, test_target)
