import sys
import os

# Set up paths
sys.path.append(os.path.join(os.path.dirname(__file__), "model"))
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

import numpy as np
from layers import forward_pass
from metrics import confusion_matrix, get_metrics, get_averages
from visualization import plot_confusion_matrix, plot_misclassified

# Class mapping: index 0-8 = digits 1-9, index 9-34 = letters A-Z
CLASS_LABELS = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "J", "K", "L", "M", "N", "O", "P", "Q", "R",
    "S", "T", "U", "V", "W", "X", "Y", "Z"
]

# Paths
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
report_dir = os.path.join(os.path.dirname(__file__), "..", "report")
figures_dir = os.path.join(report_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

# Load test data and trained parameters
test_images = np.load(os.path.join(data_dir, "processed", "test_images.npy"))
test_labels = np.load(os.path.join(data_dir, "processed", "test_labels.npy"))
parameters = np.load(os.path.join(data_dir, "trained", "parameters.npy"), allow_pickle=True).item()

print("Test images:", test_images.shape)
print("Test labels:", test_labels.shape)

# Forward pass on test set
output, _ = forward_pass(test_images, parameters)

# Get predictions and true labels (integer indices)
preds = np.argmax(output, axis=0)
true = np.argmax(test_labels, axis=0)

# Overall accuracy
accuracy = np.mean(preds == true) * 100
print(f"\nTest Accuracy: {accuracy:.2f}%")

# Confusion matrix
cm = confusion_matrix(true, preds)

# Per-class metrics
per_class = get_metrics(cm)

# Macro and weighted averages
macro, weighted = get_averages(per_class)

# Print metrics table
print("\n" + "=" * 55)
print(f"{'Class':>6} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 55)
for c in sorted(per_class.keys()):
    m = per_class[c]
    label = CLASS_LABELS[c] if c < len(CLASS_LABELS) else str(c)
    print(f"{label:>6} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
print("-" * 55)
print(f"{'Macro':>6} {macro['precision']:>10.4f} {macro['recall']:>10.4f} {macro['f1']:>10.4f}")
print("=" * 55)

# Save metrics table to report/
metrics_path = os.path.join(report_dir, "metrics.txt")
with open(metrics_path, "w") as f:
    f.write(f"Test Accuracy: {accuracy:.2f}%\n\n")
    f.write(f"{'Class':>6} {'Precision':>10} {'Recall':>10} {'F1':>10}\n")
    f.write("-" * 55 + "\n")
    for c in sorted(per_class.keys()):
        m = per_class[c]
        label = CLASS_LABELS[c] if c < len(CLASS_LABELS) else str(c)
        f.write(f"{label:>6} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}\n")
    f.write("-" * 55 + "\n")
    f.write(f"{'Macro':>6} {macro['precision']:>10.4f} {macro['recall']:>10.4f} {macro['f1']:>10.4f}\n")
print(f"\nMetrics saved to {metrics_path}")

# Save confusion matrix figure
cm_path = os.path.join(figures_dir, "confusion_matrix.png")
plot_confusion_matrix(cm, class_labels=CLASS_LABELS, save_path=cm_path)

# Save misclassified samples figure
mis_path = os.path.join(figures_dir, "misclassified.png")
plot_misclassified(test_images, true, preds, probs=output, class_labels=CLASS_LABELS, save_path=mis_path)

print("\nEvaluation complete!")
