import numpy as np
import matplotlib.pyplot as plt

# Show confusion matrix
def plot_confusion_matrix(cm, class_labels=None, save_path=None):
    num_classes = cm.shape[0]
    if class_labels is None:
        class_labels = [str(i) for i in range(num_classes)]

    # Convert counts to percentages
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    cm_percent = cm / row_sum

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_percent, cmap="Blues")
    plt.xticks(range(num_classes), class_labels, rotation=45, ha="right")
    plt.yticks(range(num_classes), class_labels)
    # Write the count inside each cell
    for i in range(num_classes):
        for j in range(num_classes):
            if cm[i, j] > 0:
                plt.text(j,i,cm[i, j],ha="center",va="center",fontsize=5)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.colorbar(label="Percentage")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print("Confusion matrix saved!")
    else:
        plt.show()
    plt.close()

# Show images that were classified incorrectly
def plot_misclassified(images, true, predicted, probs=None, num_show=25, class_labels=None, save_path=None):
    # Find incorrect predictions
    wrong = np.where(true != predicted)[0]
    if len(wrong) == 0:
        print("No wrong predictions!")
        return
    wrong = wrong[:num_show]
    
    # Create a grid
    columns = 5
    rows = int(np.ceil(len(wrong) / columns))

    plt.figure(figsize=(10, rows * 2))
    
    for i, index in enumerate(wrong):
        # Convert 784 pixels back to 28x28 (.T fixes EMNIST transpose)
        image = images[:, index].reshape(28, 28).T

        plt.subplot(rows, columns, i + 1)
        plt.imshow(image, cmap="gray")
        plt.axis("off")

        true_label = true[index]
        predicted_label = predicted[index]

        if class_labels:
            true_label = class_labels[true_label]
            predicted_label = class_labels[predicted_label]

        title = "True: " + str(true_label)
        title += "\nPred: " + str(predicted_label)
        if probs is not None:
            confidence = probs[predicted[index], index] * 100
            title +="\n" + str(round(confidence)) + "%"
        plt.title(title, fontsize=8)
    plt.suptitle("Misclassified Images")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print("Misclassified images saved!")
    else:
        plt.show()

    plt.close()