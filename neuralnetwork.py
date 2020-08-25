import numpy as np
from utils import *

"""
todo
since i went with the Laer class route, see how to make the first layer hold the data matrix while keeping initRandomWeights() untouched
i have brought this on myself...
"""


class Layer:
    def __init__(self, size=None, activation=None, data=None):
        self.size = size
        self.z_values = np.zeros(size)
        self.a_values = None
        if data is not None:
            self.a_values = data
        if activation != None:
            self.activation = activation_dict[activation]

    def computeLayer(self, prev_layer, weights):
        print(prev_layer.size, weights.shape)
        self.z_values = np.dot(prev_layer.a_values, weights.T)
        self.a_values  = activation_function(self.z_values)


class NeuralNetwork:
    def __init__(self, layers, cost_function='cross_entropy'): 
        self.layers = layers 
        self.n_input = layers[0].size # input layer size (number of feature excluding the bias)
        self.n_ouput = layers[-1].size # output layer size
        self.weights = []
        self.cost = cost_dict[cost_function]

    def initRandomWeights(self):
        # bias is included
        layers_weight_matrix_dim = [(self.layers[i].size, self.layers[i-1].size+1) for i in range(1, len(self.layers))]
        for dim in layers_weight_matrix_dim:
            self.weights.append(np.random.random(dim))

        print('initialized random weights of size:', [w.shape for w in self.weights])
    
    def train(self, X, y, learning_rate=0.01, epochs=1, regularization_lambda=0):
        # add bias to input
        X = np.c_[np.ones((X.shape[0], 1)), X]
        m = y.size
        Y = np.zeros((m, self.n_ouput))

        # sparse output vector to matrix
        for i in range(m):
            Y[i][y[i]] = 1

        # TRAINING
        for i in range(0, epochs):
            for i, layer in enumerate(self.layers):
                if i != 0:
                    layer.computeLayer(self.layers[i-1], self.weights[i-1])
            
            # error calculation
            J = self.cost(self.layers[-1], Y)
            print(J)

            

                # compute cost (with regularization) and append to an array
                # threshold stopping (if not using iterations)
            # backpropagation
                # use code from MATLAB
                # consider the activation function derivative of each layer



    