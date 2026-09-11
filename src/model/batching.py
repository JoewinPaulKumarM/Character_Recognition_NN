import numpy as np

def get_minibatches(X, Y, batch_size, seed=None):
    """Shuffle data and split into mini-batches. Seed changes each epoch
    so the batches are different every time, but reproducible."""
    # Number of training examples
    m = X.shape[1]
    # Set seed if given
    if seed is not None:
        np.random.seed(seed)
    # Shuffle the data
    indexes = np.random.permutation(m)
    X = X[:, indexes]
    Y = Y[:, indexes]
    batches = []
    # Create batches
    for start in range(0, m, batch_size):
        end = start + batch_size
        X_batch = X[:, start:end]
        Y_batch = Y[:, start:end]
        batches.append((X_batch, Y_batch))
    return batches