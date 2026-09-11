import numpy as np
from activations import relu_activation, softmax_activation, relu_derivative
# Number of neurons in each layer
layer_dims = [784, 128, 64, 35]

# Initialize weights and biases
def initialize_parameters(layer_dims, seed=56):
    np.random.seed(seed)
    parameters = {}
    # Go through each layer
    for l in range(1, len(layer_dims)):
        input_size = layer_dims[l - 1]
        output_size = layer_dims[l]
        # Weight values
        if l < len(layer_dims) - 1:
            # He initialization for ReLU layers
            parameters["W" + str(l)] = (
                np.random.randn(output_size, input_size)
                * np.sqrt(2 / input_size)
            )
        else:
            # Xavier initialization for output layer
            parameters["W" + str(l)] = (
                np.random.randn(output_size, input_size)
                * np.sqrt(1 / input_size)
            )
        # Bias starts at zero
        parameters["b" + str(l)] = np.zeros((output_size, 1))
    return parameters

# Forward propagation
def forward_pass(X, parameters):
    A = X
    # Number of layers
    L = len(parameters) // 2

    # Save values needed for backward pass
    cache = {}
    cache["A0"] = X
    # Go through all layers
    for l in range(1, L + 1):
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        # Calculate Z
        Z = np.dot(W, A) + b
        cache["Z" + str(l)] = Z
        # Hidden layers use ReLU
        if l < L:
            A = relu_activation(Z)
            cache["A" + str(l)] = A
        # Last layer uses Softmax
        else:
            A = softmax_activation(Z)
            cache["A" + str(l)] = A

    return A, cache

# Backward propagation
def backward_pass(Y, parameters, cache):
    m = Y.shape[1]
    # Number of layers
    L = len(parameters) // 2
    gradients = {}
    # Start from output layer
    dZ = cache["A" + str(L)] - Y
    # Gradient of output layer weights
    gradients["dW" + str(L)] = np.dot(dZ,cache["A" + str(L - 1)].T) / m
    # Gradient of output layer bias
    gradients["db" + str(L)] = np.sum(dZ, axis=1, keepdims=True) / m
    
    # Go backwards through hidden layers
    for l in range(L - 1, 0, -1):
        # Move error backwards
        dA = np.dot(parameters["W" + str(l + 1)].T,dZ)
        # Apply ReLU derivative
        dZ = dA * relu_derivative(cache["Z" + str(l)])
        # Gradient of weights
        gradients["dW" + str(l)] = (np.dot(dZ,cache["A" + str(l - 1)].T))/m

        # Gradient of bias
        gradients["db" + str(l)] = (
            np.sum(dZ, axis=1, keepdims=True) / m
        )
    return gradients