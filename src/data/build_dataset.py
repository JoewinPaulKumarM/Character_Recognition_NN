import sys, os
sys.path.append(os.path.dirname(__file__))

import numpy as np
from loader import load_csv
from preprocess import preprocess
from split import train_val_test_split

# Output directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
output_dir = os.path.join(project_root, "data", "processed")
os.makedirs(output_dir, exist_ok=True)

# Load raw CSV
images, labels = load_csv()

# Preprocess
images, labels = preprocess(images, labels)

# Split
train_images, train_labels, val_images, val_labels, test_images, test_labels = train_val_test_split(images, labels)

# Save all six arrays
np.save(os.path.join(output_dir, "train_images.npy"), train_images)
np.save(os.path.join(output_dir, "train_labels.npy"), train_labels)
np.save(os.path.join(output_dir, "val_images.npy"), val_images)
np.save(os.path.join(output_dir, "val_labels.npy"), val_labels)
np.save(os.path.join(output_dir, "test_images.npy"), test_images)
np.save(os.path.join(output_dir, "test_labels.npy"), test_labels)

print("\nSaved to", output_dir)
print("  train_images.npy", train_images.shape)
print("  train_labels.npy", train_labels.shape)
print("  val_images.npy  ", val_images.shape)
print("  val_labels.npy  ", val_labels.shape)
print("  test_images.npy ", test_images.shape)
print("  test_labels.npy ", test_labels.shape)
