import csv
import os
import numpy as np

# Setting up file paths
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

file_path = os.path.join(project_path, "data", "raw", "emnist-mnist-final.csv")

def load_csv(filepath=file_path):

    if not os.path.exists(filepath):
        raise FileNotFoundError("CSV file not found!")

    rows = []
    with open(filepath, newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # skip header row
        for row in reader:
            rows.append([int(x) for x in row])  # converts it into an integer   
    data = np.array(rows)
    labels = data[:, 0]
    images = data[:, 1:]
    print(np.unique(labels))
    values, counts = np.unique(labels, return_counts=True)
    print(dict(zip(values, counts)))
    return images, labels

if __name__ == "__main__":

    images, labels = load_csv()

    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
   