from neuralnetwork import *
import numpy as np

X = np.array([[5,3,2],[45,2,31],[4,5,16],[1,1,6],[44,56,0]])
y = np.array([0,1])
n_input = X.shape[1]
print(n_input)

nn = NeuralNetwork([n_input, 3, 2])
nn.initRandomWeights()
nn.train(X, y)
#nn.summary()
