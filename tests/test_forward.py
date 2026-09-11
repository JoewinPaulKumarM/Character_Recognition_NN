import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "data"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "model"))

import numpy as np
from loader import load_csv
from preprocess import preprocess
from split import train_val_test_split
from layers import initialize_parameters, forward_pass

# Load, preprocess, split
images, labels = load_csv()
images, labels = preprocess(images, labels)
train_images, train_labels, val_images, val_labels, test_images, test_labels = train_val_test_split(images, labels)

# Initialize network
layer_dims = [784, 128, 64, 35]
params = initialize_parameters(layer_dims)

# Forward pass on training data
output, cache = forward_pass(train_images, params)

print("\n--- Forward Pass Tests ---")

# 1. Output shape
expected_shape = (35, train_images.shape[1])
assert output.shape == expected_shape, f"Output shape {output.shape} != expected {expected_shape}"
print(f"[PASS] Output shape: {output.shape}")

# 2. Softmax probabilities sum to 1
col_sums = np.sum(output, axis=0)
assert np.allclose(col_sums, 1.0), f"Softmax sums not 1.0: min={col_sums.min()}, max={col_sums.max()}"
print(f"[PASS] Softmax sums to 1.0 (min={col_sums.min():.6f}, max={col_sums.max():.6f})")

# 3. All values between 0 and 1
assert output.min() >= 0, f"Negative output found: {output.min()}"
assert output.max() <= 1, f"Output > 1 found: {output.max()}"
print(f"[PASS] All outputs in [0, 1] (min={output.min():.6f}, max={output.max():.6f})")

# 4. Cache contains all expected keys
L = len(params) // 2
expected_keys = ["A0"]
for l in range(1, L + 1):
    expected_keys.append("Z" + str(l))
    expected_keys.append("A" + str(l))
missing = [k for k in expected_keys if k not in cache]
assert len(missing) == 0, f"Missing cache keys: {missing}"
print(f"[PASS] All {len(expected_keys)} cache keys present: {sorted(cache.keys())}")

# 5. Cache shapes are correct
assert cache["A0"].shape == train_images.shape, "A0 shape mismatch"
assert cache["Z1"].shape == (128, train_images.shape[1]), "Z1 shape mismatch"
assert cache["A1"].shape == (128, train_images.shape[1]), "A1 shape mismatch"
assert cache["Z2"].shape == (64, train_images.shape[1]), "Z2 shape mismatch"
assert cache["A2"].shape == (64, train_images.shape[1]), "A2 shape mismatch"
assert cache["Z3"].shape == (35, train_images.shape[1]), "Z3 shape mismatch"
assert cache["A3"].shape == (35, train_images.shape[1]), "A3 shape mismatch"
print("[PASS] All cache shapes correct")

# 6. ReLU outputs are non-negative
assert cache["A1"].min() >= 0, "ReLU layer 1 has negative values"
assert cache["A2"].min() >= 0, "ReLU layer 2 has negative values"
print("[PASS] ReLU outputs are non-negative")

print("\nAll forward pass tests passed!")
