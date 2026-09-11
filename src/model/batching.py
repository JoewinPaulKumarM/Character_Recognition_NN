import numpy as np

def get_minibatches(X, Y, batch_size, seed=None):
    m = X.shape[1]
    minibatches = []

    if seed is not None:
        np.random.seed(seed)

    # Shuffle columns together
    permutation = np.random.permutation(m)
    X_shuffled = X[:, permutation]
    Y_shuffled = Y[:, permutation]

    # Full batches
    num_full = m // batch_size
    for k in range(num_full):
        start = k * batch_size
        end = start + batch_size
        minibatches.append((X_shuffled[:, start:end], Y_shuffled[:, start:end]))

    # Leftover batch (if any)
    if m % batch_size != 0:
        minibatches.append((X_shuffled[:, num_full * batch_size:], Y_shuffled[:, num_full * batch_size:]))

    return minibatches
