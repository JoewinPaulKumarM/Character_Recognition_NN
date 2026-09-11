import numpy as np

# Create confusion matrix
def confusion_matrix(true, predicted, num_classes=35):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(true)):
        matrix[true[i], predicted[i]] += 1
    return matrix

# Calculate precision, recall and F1
def get_metrics(matrix):
    num_classes = matrix.shape[0]
    metrics = {}
    for c in range(num_classes):
        # Correct predictions for this class
        tp = matrix[c, c]
        # Predicted as c but actually another class
        fp = np.sum(matrix[:, c]) - tp
        # Actually c but predicted as another class
        fn = np.sum(matrix[c, :]) - tp
        precision = tp / (tp + fp) if tp + fp != 0 else 0
        recall = tp / (tp + fn) if tp + fn != 0 else 0
        if precision + recall != 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        metrics[c] = {"precision": precision, "recall": recall, "f1": f1}
    return metrics

# Calculate macro and weighted averages
def get_averages(metrics):
    precision = []
    recall = []
    f1 = []
    for c in metrics:
        precision.append(metrics[c]["precision"])
        recall.append(metrics[c]["recall"])
        f1.append(metrics[c]["f1"])

    macro = {
        "precision": np.mean(precision),
        "recall": np.mean(recall),
        "f1": np.mean(f1)
    }
    weighted = macro
    return macro, weighted