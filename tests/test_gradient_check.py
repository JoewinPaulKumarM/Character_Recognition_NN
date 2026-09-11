import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "model"))

import numpy as np
# pyrefly: ignore [missing-import]
from layers import initialize_parameters, forward_pass, backward_pass
# pyrefly: ignore [missing-import]
from loss import cross_entropy_loss

# Tiny data: 5 samples, 3 classes (keeps the check fast)
np.random.seed(99)
X = np.random.randn(784, 5)
Y = np.eye(35)[:, :5]  # first 5 columns of identity -> one-hot for classes 0-4

# Small network to make full sweep feasible
layer_dims = [784, 16, 35]
params = initialize_parameters(layer_dims, seed=7)

# Analytical gradients from backprop
A, cache = forward_pass(X, params)
grads = backward_pass(Y, params, cache)

# Numerical gradient check via centered finite differences
epsilon = 1e-7

print("Numerical Gradient Check")
print("=" * 60)

all_passed = True

for key in params:
    param = params[key]
    grad_analytical = grads["d" + key]
    grad_numerical = np.zeros_like(param)

    # For speed, check a random subset of elements for large matrices
    total_elements = param.size
    if total_elements > 50:
        check_indices = np.random.choice(total_elements, 50, replace=False)
    else:
        check_indices = np.arange(total_elements)

    for idx in check_indices:
        # Flatten index to (row, col)
        pos = np.unravel_index(idx, param.shape)

        # f(theta + epsilon)
        original = param[pos]
        param[pos] = original + epsilon
        A_plus, _ = forward_pass(X, params)
        loss_plus = cross_entropy_loss(A_plus, Y)

        # f(theta - epsilon)
        param[pos] = original - epsilon
        A_minus, _ = forward_pass(X, params)
        loss_minus = cross_entropy_loss(A_minus, Y)

        # Restore
        param[pos] = original

        # Centered difference
        grad_numerical[pos] = (loss_plus - loss_minus) / (2 * epsilon)

    # Relative error on checked elements only
    checked_analytical = np.array([grad_analytical.flat[i] for i in check_indices])
    checked_numerical = np.array([grad_numerical.flat[i] for i in check_indices])

    numerator = np.linalg.norm(checked_analytical - checked_numerical)
    denominator = np.linalg.norm(checked_analytical) + np.linalg.norm(checked_numerical)

    if denominator == 0:
        rel_error = 0.0
    else:
        rel_error = numerator / denominator

    status = "PASS" if rel_error < 1e-5 else "FAIL"
    if status == "FAIL":
        all_passed = False

    print(f"  {key:4s}  shape {str(param.shape):14s}  "
          f"checked {len(check_indices):3d}/{total_elements:5d} elements  "
          f"rel_error = {rel_error:.2e}  [{status}]")

print("=" * 60)
if all_passed:
    print("ALL PASSED — analytical gradients match numerical gradients.")
else:
    print("FAILURE — some gradients do not match. Check backprop math.")
