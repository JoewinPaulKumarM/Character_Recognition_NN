import sys
import os

# Find model and utility files
sys.path.append(os.path.join(os.path.dirname(__file__), "model"))
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

import numpy as np

from layers import forward_pass
from metrics import confusion_matrix, get_metrics, get_averages
from visualization import plot_confusion_matrix, plot_misclassified


# Labels for the 35 classes
CLASS_LABELS = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "J", "K", "L", "M", "N", "O", "P", "Q", "R",
    "S", "T", "U", "V", "W", "X", "Y", "Z"
]

# Folders
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
report_dir = os.path.join(os.path.dirname(__file__), "..", "report")
figures_dir = os.path.join(report_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)
# Load test data
test_images = np.load(os.path.join(data_dir, "processed", "test_images.npy"))
test_labels = np.load(os.path.join(data_dir, "processed", "test_labels.npy"))
# Load trained network
parameters = np.load(os.path.join(data_dir, "trained", "parameters.npy"),allow_pickle=True).item()

print("Test images:", test_images.shape)
print("Test labels:", test_labels.shape)

# Make predictions
output, _ = forward_pass(test_images, parameters)

predictions = np.argmax(output, axis=0)
actual = np.argmax(test_labels, axis=0)
# Calculate accuracy
accuracy = np.mean(predictions == actual) * 100
print("\nTest Accuracy:", round(accuracy, 2), "%")
# Confusion matrix
cm = confusion_matrix(actual, predictions)
# Precision, recall and F1
metrics = get_metrics(cm)
macro, weighted = get_averages(metrics)

# Print results
print("\nClass    Precision    Recall    F1")
for c in metrics:
    label = CLASS_LABELS[c]
    print(label,"     ",round(metrics[c]["precision"], 4),"      ",round(metrics[c]["recall"], 4),"    ",round(metrics[c]["f1"], 4))

print("\nMacro Precision:", round(macro["precision"], 4))
print("Macro Recall:", round(macro["recall"], 4))
print("Macro F1:", round(macro["f1"], 4))


# Save metrics to a text file
metrics_file = os.path.join(
    report_dir, "metrics.txt"
)

with open(metrics_file, "w") as file:
    file.write("Test Accuracy: "+ str(round(accuracy, 2))+ "%\n\n")
    file.write("Class    Precision    Recall    F1\n")
    for c in metrics:
        label = CLASS_LABELS[c]
        file.write(label+ "    "+ str(round(metrics[c]["precision"], 4))+ "    "+ str(round(metrics[c]["recall"], 4))+ "    "+ str(round(metrics[c]["f1"], 4))+ "\n")

    file.write("\nMacro Precision: " + str(round(macro["precision"], 4)))
    file.write("\nMacro Recall: " + str(round(macro["recall"], 4)))
    file.write("\nMacro F1: " + str(round(macro["f1"], 4)))

print("\nMetrics saved to:", metrics_file)

# Save confusion matrix
confusion_file = os.path.join(figures_dir, "confusion_matrix.png")
plot_confusion_matrix(cm,class_labels=CLASS_LABELS,save_path=confusion_file)

# Save wrong predictions
wrong_file = os.path.join(figures_dir, "misclassified.png")
plot_misclassified(test_images,actual,predictions,probs=output,class_labels=CLASS_LABELS,save_path=wrong_file)
print("\nEvaluation complete!")