## This code can be used to create, train and query 3-layer neural networks 

import numpy as np
# scipy.special for the sigmoid function expit()
import scipy.special


# neural network class definition
class neuralNetwork:
    # initialize the neural network
    def __init__(self, inputNodes, hiddenNodes, outputNodes,learningRate):
        # set number of nodes in each input, hidden and output layer
        self.inodes = inputNodes
        self.hnodes = hiddenNodes
        self.onodes = outputNodes
        
        # learning rate
        self.lrate = learningRate
        
        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j,  where link is form node i to node j in the next layer
        #self.wih = (np.random.rand(self.hnodes, self.inodes)-0.5) 
        #self.who = (np.random.rand(self.onodes,self.hnodes)-0.5)
        
        # here is another approach to initialize the weights
        self.wih = (np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes)))
        self.who = (np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes)))
        
        # define the activation function
        #self.activation_function = lambda x: scipy.special.expit(x)
        # pickle cannot serialize the lambda function
    def activation_function(self,x):
        return scipy.special.expit(x)
    # train the neural network
    def train(self,inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list,ndmin=2).T
        targets = np.array(targets_list,ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih,inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # compute error
        output_errors = targets - final_outputs
        # back-propagated errors for the hidden layer nodes, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)


        # update the weights for the links between the hidden and output layers
        self.who += self.lrate * np.dot(output_errors * final_outputs *(1 - final_outputs), np.transpose(hidden_outputs))
        # update the weights for the links between the input and hidden layers
        self.wih += self.lrate * np.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs), np.transpose(inputs))
        
        pass

    # query the neural network 
    def query(self, input_list):
        # convert inputs list to 2d array
        inputs = np.array(input_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

