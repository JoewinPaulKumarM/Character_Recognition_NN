import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src", "model"))
import numpy as np
from layers import forward_pass

params = np.load('data/trained/parameters.npy', allow_pickle=True).item()
val_images = np.load('data/processed/val_images.npy')
val_labels = np.load('data/processed/val_labels.npy')

A, _ = forward_pass(val_images, params)
preds = np.argmax(A, axis=0)
true = np.argmax(val_labels, axis=0)
print("Val accuracy:", np.mean(preds == true) * 100, "%")