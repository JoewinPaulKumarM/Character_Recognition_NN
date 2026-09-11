import numpy as np
import matplotlib.pyplot as plt

val_images = np.load('data/processed/val_images.npy')
val_labels = np.load('data/processed/val_labels.npy')
true = np.argmax(val_labels, axis=0)

for idx in [0, 8, 9, 34]:
    pos = np.where(true == idx)[0][0]
    img = val_images[:, pos].reshape(28, 28).T
    plt.imshow(img, cmap='gray')
    plt.title(f"index {idx}")
    plt.show()