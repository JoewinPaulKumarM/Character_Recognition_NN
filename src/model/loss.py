import numpy as np
def cross_entropy_loss(Y_hat, Y):
    m = Y.shape[1]
    # Avoid log(0)
    Y_hat = np.clip(Y_hat, 1e-12, 1)
    # Calculate loss
    loss = -np.sum(Y * np.log(Y_hat)) / m
    return loss