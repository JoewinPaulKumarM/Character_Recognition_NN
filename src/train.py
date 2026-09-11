import sys
import os

# Find files inside the model folder
sys.path.append(os.path.join(os.path.dirname(__file__), "model"))

import numpy as np

from layers import initialize_parameters, forward_pass, backward_pass
from loss import cross_entropy_loss
from optimizers import initialize_adam, update_adam
from batching import get_minibatches


# Load processed data
data_dir = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "processed"
)

train_images = np.load(os.path.join(data_dir, "train_images.npy"))
train_labels = np.load(os.path.join(data_dir, "train_labels.npy"))

val_images = np.load(os.path.join(data_dir, "val_images.npy"))
val_labels = np.load(os.path.join(data_dir, "val_labels.npy"))

print("Train:", train_images.shape, train_labels.shape)
print("Validation:", val_images.shape, val_labels.shape)


# Network settings
layer_dims = [784, 128, 64, 35]

learning_rate = 0.001
epochs = 50
batch_size = 64


# Initialize network
parameters = initialize_parameters(layer_dims)

# Initialize Adam
v, s = initialize_adam(parameters)
t = 0

# History tracking
history = {"loss": [], "train_acc": [], "val_acc": []}

# Training
for epoch in range(epochs):

    total_loss = 0
    batch_count = 0

    # Create mini-batches
    minibatches = get_minibatches(
        train_images,
        train_labels,
        batch_size,
        seed=epoch
    )

    # Train using each batch
    for X_batch, Y_batch in minibatches:

        # Forward pass
        output, cache = forward_pass(
            X_batch,
            parameters
        )

        # Calculate loss
        loss = cross_entropy_loss(
            output,
            Y_batch
        )

        total_loss += loss
        batch_count += 1

        # Backward pass
        gradients = backward_pass(
            Y_batch,
            parameters,
            cache
        )

        # Update weights using Adam
        t += 1

        parameters, v, s = update_adam(
            parameters,
            gradients,
            v,
            s,
            t,
            learning_rate
        )


    # Average loss for this epoch
    average_loss = total_loss / batch_count


    # Validation accuracy
    val_output, _ = forward_pass(
        val_images,
        parameters
    )

    val_predictions = np.argmax(val_output, axis=0)
    val_true = np.argmax(val_labels, axis=0)

    val_accuracy = np.mean(
        val_predictions == val_true
    ) * 100


    # Training accuracy
    train_output, _ = forward_pass(
        train_images,
        parameters
    )

    train_predictions = np.argmax(
        train_output,
        axis=0
    )

    train_true = np.argmax(
        train_labels,
        axis=0
    )

    train_accuracy = np.mean(
        train_predictions == train_true
    ) * 100


    # Show results
    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Loss: {average_loss:.4f} | "
        f"Train: {train_accuracy:.1f}% | "
        f"Val: {val_accuracy:.1f}%"
    )

    # Save history
    history["loss"].append(average_loss)
    history["train_acc"].append(train_accuracy)
    history["val_acc"].append(val_accuracy)

# Save trained parameters
save_dir = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "trained"
)

os.makedirs(save_dir, exist_ok=True)

np.save(
    os.path.join(save_dir, "parameters.npy"),
    parameters
)

np.save(
    os.path.join(save_dir, "history.npy"),
    history
)

print("\nTraining complete!")
print("Parameters and history saved to data/trained/")