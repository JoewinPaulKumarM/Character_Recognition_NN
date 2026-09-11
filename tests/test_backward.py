import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "data"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "model"))

import numpy as np

from loader import load_csv
from preprocess import preprocess
from split import train_val_test_split
from layers import initialize_parameters, forward_pass, backward_pass
from loss import cross_entropy_loss

# Load, preprocess, split
images, labels = load_csv()
images, labels = preprocess(images, labels)
train_images, train_labels, val_images, val_labels, test_images, test_labels = train_val_test_split(images, labels)

# Initialize network
layer_dims = [784, 128, 64, 35]
params = initialize_parameters(layer_dims)

# Forward pass
output, cache = forward_pass(train_images, params)

# Backward pass
grads = backward_pass(train_labels, params, cache)

print("\n--- Backward Pass Tests ---")

# 1. Shape consistency: every gradient matches its parameter
L = len(params) // 2
all_match = True
for l in range(1, L + 1):
    for key in ["W", "b"]:
        p_shape = params[key + str(l)].shape
        g_shape = grads["d" + key + str(l)].shape
        if p_shape != g_shape:
            all_match = False
            print(f"[FAIL] {key}{l}: param {p_shape} != grad {g_shape}")
        else:
            print(f"[PASS] d{key}{l} shape {g_shape} matches {key}{l} {p_shape}")
assert all_match, "Shape mismatch found!"

# 2. No NaN or Inf in gradients
print()
for l in range(1, L + 1):
    for key in ["W", "b"]:
        g = grads["d" + key + str(l)]
        assert not np.any(np.isnan(g)), f"NaN in d{key}{l}"
        assert not np.any(np.isinf(g)), f"Inf in d{key}{l}"
print("[PASS] No NaN or Inf in any gradient")

# 3. Gradients are not all zero (network is learning something)
for l in range(1, L + 1):
    dW = grads["dW" + str(l)]
    assert np.any(dW != 0), f"dW{l} is all zeros"
print("[PASS] No all-zero weight gradients")

# 4. Loss decreases after one update step
loss_before = cross_entropy_loss(output, train_labels)

lr = 0.1
updated_params = {}
for l in range(1, L + 1):
    updated_params["W" + str(l)] = params["W" + str(l)] - lr * grads["dW" + str(l)]
    updated_params["b" + str(l)] = params["b" + str(l)] - lr * grads["db" + str(l)]

output_after, _ = forward_pass(train_images, updated_params)
loss_after = cross_entropy_loss(output_after, train_labels)

print(f"\n[INFO] Loss before update: {loss_before:.6f}")
print(f"[INFO] Loss after update:  {loss_after:.6f}")
assert loss_after < loss_before, f"Loss did not decrease: {loss_before:.6f} -> {loss_after:.6f}"
print("[PASS] Loss decreased after one gradient step")

print("\nAll backward pass tests passed!")
