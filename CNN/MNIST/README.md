# MNIST Digit Classification

A convolutional neural network for classifying handwritten digits from the MNIST dataset.

## Dataset

`train.csv` contains flattened 28x28 grayscale images. The `label` column contains the digit class from 0 to 9; the remaining 784 columns are pixel values.

## Workflow

The notebook:

1. Loads and inspects the training data.
2. Splits the data into training, development, and test sets.
3. Standardizes the pixel features using statistics from the training set.
4. Reshapes each image to `(28, 28, 1)` for the CNN.
5. Trains a TensorFlow/Keras model with two convolution and pooling blocks, followed by dense layers.
6. Evaluates test loss and accuracy.

## Model

- Conv2D: 32 filters
- MaxPooling2D
- Conv2D: 64 filters
- MaxPooling2D
- Flatten
- Dense: 128 units with ReLU
- Dense: 10 units with softmax

## Run

Open `model.ipynb` in Jupyter or VS Code and run the cells from top to bottom. Install the required packages in the active Python environment first:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

The notebook expects `train.csv` to be in the same directory as the notebook.
