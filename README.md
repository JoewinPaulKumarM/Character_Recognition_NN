# Character Recognition Neural Network from Scratch

A 35-class handwritten character recognition neural network built entirely
from scratch using **Python and NumPy**.

THE REPORT FILE PDF IS LOCATED INSIDE report folder contains all information about the project
Project output images:https://drive.google.com/drive/folders/1W3iASSd59-m1T1fN9Sv7lAvcjqfziSyR?usp=sharing
## Objective

Classify handwritten:

- Uppercase letters: A-Z
- Digits: 1-9
- Total: 35 classes

No TensorFlow, PyTorch, Keras, or other ML frameworks were used.

## Dataset

- 10,500 images
- 300 samples per class
- Image size: 28 × 28
- 35 balanced classes
- Train / Validation / Test: 70% / 15% / 15%

## Model

```text
784 → 128 → 64 → 35
