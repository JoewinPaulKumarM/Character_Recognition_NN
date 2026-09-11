import numpy as np

def train_val_test_split(images, labels, seed=42):
    """Stratified 70/15/15 split. Splits each class independently so every
    class has proportional representation in train, val, and test."""
    # Get class number from one-hot labels
    class_labels = np.argmax(labels, axis=0)
    rng = np.random.default_rng(seed)
    train_index = []
    val_index = []
    test_index = []

    # Do the split for each class
    for c in range(35):
        # Find all images of this class
        indexes = np.where(class_labels == c)[0]
        # Shuffle the indexes
        rng.shuffle(indexes)
        # Find how many go into each group
        n = len(indexes)
        train_count = int(n * 0.70)
        val_count = int(n * 0.15)
        # Add indexes
        train_index.extend(indexes[:train_count])
        val_index.extend(indexes[train_count:train_count + val_count])
        test_index.extend(indexes[train_count + val_count:])

    # Shuffle the final groups
    np.random.shuffle(train_index)
    np.random.shuffle(val_index)
    np.random.shuffle(test_index)

    # Get the actual data
    train_images = images[:,train_index]
    train_labels = labels[:,train_index]

    val_images = images[:,val_index]
    val_labels = labels[:,val_index]

    test_images = images[:,test_index]
    test_labels = labels[:,test_index]

    print("Training data:", train_images.shape)
    print("Validation data:", val_images.shape)
    print("Testing data:", test_images.shape)

    # check number of samples in each class
    print("\nClass counts (Train / Validation / Test):")

    train_classes = np.argmax(train_labels, axis=0)
    val_classes = np.argmax(val_labels, axis=0)
    test_classes = np.argmax(test_labels, axis=0)

    for c in range(35):
        train_count = np.sum(train_classes == c)
        val_count = np.sum(val_classes == c)
        test_count = np.sum(test_classes == c)  

        print("Class", c+1,":", train_count,"/", val_count,"/", test_count)

    return train_images, train_labels, val_images, val_labels, test_images, test_labels 

# Test
if __name__ == "__main__":
    from loader import load_csv
    from preprocess import preprocess
    images, labels = load_csv()
    images, labels = preprocess(images, labels)
    train_images, train_labels, val_images, val_labels, test_images, test_labels = train_val_test_split(
        images, labels
    )