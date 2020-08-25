
# activation functions

def sigmoid(z):
    return 1/(1+np.exp(-z))

def tanh(z):
    return np.tanh(z)

def relu(z):
    return z * (z > 0)

activation_dict = {'sigmoid':sigmoid, 'tanh': tanh, 'relu':relu}

# activation function derivatives
def d_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))

def d_tanh(z):
    return 1-np.tanh(z)**2

def d_relu(z):
    return 1 * (x > 0)

d_activation_dict = {'d_sigmoid':d_sigmoid, 'd_tanh': tanh, 'd_relu':d_relu}

# cost functions

def mse(X, y, weights):
    return np.sum(np.square(X.dot(weights)-y)) / (2*y.size)

def cross_entropy(output_layer, Y, alpha=1):
    h = output_layer.a_values
    m = h.size
    error = -(Y*np.log(h)+(1-Y)*np.log(1-h))/m;
    #NB: we remove the bias weight in the regularization term
    J = np.sum(error) #+ alpha*(np.sum()) + sum(sum(Theta2(:,2:end).^2)))/(2*m); 
    return J

cost_dict = {'mse':mse, 'cross_entropy':cross_entropy}

# cost function derivatives

def d_mse(X, y, weights):
    pass

def d_cross_entropy():
    pass