import numpy as np

# Gradient Descent
def update_gd(parameters, gradients, learning_rate):
    """Vanilla gradient descent: W -= lr * dW for each layer."""
    for l in range(1, len(parameters) // 2 + 1):
        parameters["W" + str(l)] -= (learning_rate * gradients["dW" + str(l)])
        parameters["b" + str(l)] -= (learning_rate * gradients["db" + str(l)])
    return parameters

# Momentum
def initialize_velocity(parameters):
    v = {}
    for l in range(1, len(parameters) // 2 + 1):
        v["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        v["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
    return v

def update_momentum(parameters, gradients, v, learning_rate, beta=0.9):
    """SGD with momentum. v is the exponentially weighted moving average of gradients."""
    for l in range(1, len(parameters) // 2 + 1):
        v["dW" + str(l)] = (beta * v["dW" + str(l)] + (1 - beta) * gradients["dW" + str(l)])
        v["db" + str(l)] = (beta * v["db" + str(l)] + (1 - beta) * gradients["db" + str(l)])
        parameters["W" + str(l)] -= learning_rate * v["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * v["db" + str(l)]
    return parameters, v

# RMSprop
def initialize_rmsprop(parameters):
    s = {}
    for l in range(1, len(parameters) // 2 + 1):
        s["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        s["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
    return s


def update_rmsprop(parameters, gradients, s, learning_rate, beta=0.999, epsilon=1e-8):
    """RMSprop: scales learning rate per-parameter by inverse sqrt of squared gradient history."""
    for l in range(1, len(parameters) // 2 + 1):
        s["dW" + str(l)] = (beta * s["dW" + str(l)] + (1 - beta) * gradients["dW" + str(l)] ** 2)
        s["db" + str(l)] = (beta * s["db" + str(l)] + (1 - beta) * gradients["db" + str(l)] ** 2)
        parameters["W" + str(l)] -= (learning_rate * gradients["dW" + str(l)] / (np.sqrt(s["dW" + str(l)]) + epsilon))
        parameters["b" + str(l)] -= (learning_rate * gradients["db" + str(l)] / (np.sqrt(s["db" + str(l)]) + epsilon))
    return parameters, s

# Adam
def initialize_adam(parameters):
    v = {}
    s = {}
    for l in range(1, len(parameters) // 2 + 1):
        v["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        v["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
        s["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        s["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
    return v, s


def update_adam(parameters,gradients,v,s,t,learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-8):
    """Adam optimizer. Combines momentum (v) and RMSprop (s) with bias correction.
    t is the global step count — needed because v and s are initialized at zero,
    so early updates would be biased toward zero without the 1/(1-beta^t) correction."""
    for l in range(1, len(parameters) // 2 + 1):
        # Momentum
        v["dW" + str(l)] = (beta1 * v["dW" + str(l)] + (1 - beta1) * gradients["dW" + str(l)])
        v["db" + str(l)] = (beta1 * v["db" + str(l)] + (1 - beta1) * gradients["db" + str(l)])
        # RMSprop
        s["dW" + str(l)] = (beta2 * s["dW" + str(l)] + (1 - beta2) * gradients["dW" + str(l)] ** 2)
        s["db" + str(l)] = (beta2 * s["db" + str(l)] + (1 - beta2) * gradients["db" + str(l)] ** 2)

        # Bias correction
        v_w = v["dW" + str(l)] / (1 - beta1 ** t)
        v_b = v["db" + str(l)] / (1 - beta1 ** t)

        s_w = s["dW" + str(l)] / (1 - beta2 ** t)
        s_b = s["db" + str(l)] / (1 - beta2 ** t)

        # Update
        parameters["W" + str(l)] -= (learning_rate * v_w / (np.sqrt(s_w) + epsilon))
        parameters["b" + str(l)] -= (learning_rate * v_b / (np.sqrt(s_b) + epsilon))
    
    return parameters, v, s
