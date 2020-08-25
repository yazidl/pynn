import numpy as np
from utils import *

"""
- implement a 'Layer' class
- add 'Layer' instances to the network
- 
"""

class NeuralNetwork():
    def __init__(self, layers, activation_function='sigmoid', cost_function='mse'): 
        self.layers = layers
        self.n_input = self.layers[0] # input layer siz (number of feature exccluding the bias)
        self.n_ouput = self.layers[-1] # output layer size
        self.n_weights = sum([i*j for i, j in zip(self.layers[:-1], self.layers[1:])]) # number of weights between layers
        self.weights = []
        self.activation = activation_dict[activation_function] # activation function
        self.cost = cost_dict[cost_function]

    def summary(self):
        info = {'input size': self.n_input,
                'layers': len(self.layers)-1,
                'output size': self.layers[-1],
                'number of weights' : self.n_weights,
                'activation function': self.activation.__name__,
                'cost function': self.cost.__name__}

        for k, v in info.items():
            print(k, ':', v)
    
    def initRandomWeights(self):
        # with bias
        layers_weight_matrix_dim = [(self.layers[i], self.layers[i-1]+1) for i in range(1,len(self.layers))]
        for dim in layers_weight_matrix_dim:
            self.weights.append(np.random.random(dim))
    
    def train(self, X, y, learning_rate=0.01, epochs=1, regularization_lambda=0):
        # add bias to input
        X = np.c_[np.ones((X.shape[0], 1)), X]
        m = y.size
        Y = np.zeros((m, self.n_ouput))
        for i in range(m):
            Y[i][y[i]] = 1

        # training
        for i in range(0, epochs):
            layer_z = []
            layer_a = []
            for layer in self.layers[1,:]:
                layer_z.append()
                # compute and store z and activation(z)
            # error calculation
                # compute cost (with regularization) and append to an array
                # threshold stopping (if not using iterations)
            # backpropagation
                # use code from MATLAB
                # consider the activation function derivative of each layer



    