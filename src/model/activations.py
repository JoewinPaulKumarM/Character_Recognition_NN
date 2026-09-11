import numpy as np

# implement softmax function
def softmax_activation(x):
    # prevent overflow by subtracting max value
    Z = np.exp(x - np.max(x, axis=0, keepdims=True))
    Z = Z/np.sum(Z,axis=0,keepdims=True)    
    return Z

# implement relu function
def relu_activation(x):
    Z = np.maximum(0, x)
    return Z

def relu_derivative(x):
    return (x > 0).astype(float)