from neuralnetwork import *


#X = np.array([[1,3],[2,5],[8,7]])
#y = np.array([1,0,0])

X = np.loadtxt('data/X.csv', delimiter=',')
y = np.loadtxt('data/y.csv', delimiter=',', dtype=int)

layers = [Layer(data=X),
          Layer(25, activation='sigmoid'),
          Layer(10, activation='sigmoid')]

nn = NeuralNetwork(layers)
nn.initRandomWeights()
nn.train(X, y)
