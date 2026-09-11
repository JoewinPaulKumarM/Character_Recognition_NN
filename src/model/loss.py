import numpy as np

# Cross-entropy loss
def cross_entropy_loss(Y_hat, Y):
    """Average cross-entropy loss over m samples. Lower = better predictions."""
    m = Y.shape[1]
    # Avoid log(0)
    Y_hat = np.clip(Y_hat, 1e-12, 1)
    loss = -np.sum(Y * np.log(Y_hat)) / m
    return loss

# L2 regularization
def l2_penalty(parameters, lambd, m):
    """Weight decay penalty: (lambd/2m) * sum(W^2). Only weights, not biases.
    Add this to cross_entropy_loss when logging so the reported loss matches
    the actual objective being optimized."""
    penalty = 0
    # Add squared weights
    for l in range(1, len(parameters) // 2 + 1):
        penalty += np.sum(parameters["W" + str(l)] ** 2)
    penalty = (lambd / (2 * m)) * penalty
    return penalty

# Add L2 term to weight gradients
def add_l2_to_gradients(gradients, parameters, lambd, m):
    """Add (lambd/m)*W to each dW gradient. Call after backward_pass,
    before the optimizer step. Only modifies weight gradients, not biases."""
    for l in range(1, len(parameters) // 2 + 1):
        gradients["dW" + str(l)] += ((lambd / m) * parameters["W" + str(l)])
    return gradients