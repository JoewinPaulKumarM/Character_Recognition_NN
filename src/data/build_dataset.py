import os
import numpy as np
from loader import load_csv
from preprocess import preprocess
from split import train_val_test_split

# Create processed data folder
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
output_dir = os.path.join(project_root, "data", "processed")
os.makedirs(output_dir, exist_ok=True)

# Load data
images, labels = load_csv()

# Preprocess data
images, labels = preprocess(images, labels)

# Split data
train_images, train_labels, val_images, val_labels, test_images, test_labels = train_val_test_split(
    images, labels
)

# Save data
np.save(os.path.join(output_dir, "train_images.npy"), train_images)
np.save(os.path.join(output_dir, "train_labels.npy"), train_labels)

np.save(os.path.join(output_dir, "val_images.npy"), val_images)
np.save(os.path.join(output_dir, "val_labels.npy"), val_labels)

np.save(os.path.join(output_dir, "test_images.npy"), test_images)
np.save(os.path.join(output_dir, "test_labels.npy"), test_labels)

print("\nProcessed data saved successfully!")
print("Train:", train_images.shape, train_labels.shape)
print("Validation:", val_images.shape, val_labels.shape)
print("Test:", test_images.shape, test_labels.shape)