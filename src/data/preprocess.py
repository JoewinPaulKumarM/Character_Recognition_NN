from loader import load_csv
import numpy as np

NUM_CLASSES = 35
IMAGE_SIZE = 28
# convert image to grayscale
def ensure_grayscale(images):
    if images.ndim == 4 and images.shape[-1] == 3:
        return (0.2989 * images[..., 0] + 0.5870 * images[..., 1] + 0.1140 * images[..., 2])
        
    if images.ndim == 4 and images.shape[-1] == 1:
        return images[..., 0]

    return images

# resize image to 28x28
def resize(images, size=28):
    if images.shape[1] == size and images.shape[2] == size:
        return images

    #Calculates how many pixels to skip
    step = images.shape[1] // size

    return images[:, ::step, ::step]

#convert pixels from 0-255 to 0-1
def normalize(images):
    return images.astype(float)/255.0

#convert 28x28 image into 784 values
def flatten(images):
    return images.reshape(images.shape[0], -1)

#convert labels into one-hot encoding
def one_hot_encode(labels, num_classes=35):
    """Convert integer labels (1-35) to one-hot vectors.
    Subtracts 1 because raw CSV labels start at 1, not 0."""
    labels = labels.astype(int)
    return np.eye(num_classes)[labels - 1]

# complete preprocessing
def preprocess(images, labels, num_classes=35):
    """Full preprocessing pipeline. Output shapes are (784, m) for images
    and (num_classes, m) for labels — columns are samples, which is what
    the network's matrix multiplications expect."""
    # Convert flat (m, 784) into (m, 28, 28) spatial grid
    if images.ndim == 2:
        side = int(np.sqrt(images.shape[1]))
        images = images.reshape(-1, side, side)

    # Preprocessing operations
    images = ensure_grayscale(images)
    images = resize(images)
    images = normalize(images)
    images = flatten(images)  # (m, 784)
    labels = one_hot_encode(labels, num_classes)  # (m, num_classes)

    # Transpose once at the end for network compatibility
    images = images.T  # (784, m)
    labels = labels.T  # (num_classes, m)

    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Pixel range:", images.min(), "to", images.max())

    return images, labels

if __name__ == "__main__":
    images, labels = load_csv()
    images, labels = preprocess(images, labels)