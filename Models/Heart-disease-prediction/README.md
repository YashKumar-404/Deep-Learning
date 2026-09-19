# Heart Disease Prediction

A binary classification model that predicts whether a patient has heart disease from clinical measurements.

## Dataset

`heart_disease_data.csv` contains the input health features and a `target` column. The target is used as the binary label.

## Workflow

The notebook:

1. Loads the dataset and separates features from the target.
2. Creates a stratified 80/20 train-test split.
3. Standardizes the features with `StandardScaler` fitted on the training data.
4. Trains a TensorFlow/Keras feed-forward neural network.
5. Evaluates accuracy on the training and test sets.
6. Generates binary predictions using a 0.5 probability threshold.

## Model

The neural network uses dense layers with 32, 16, 8, 4, and 2 ReLU units, followed by a sigmoid output for binary classification. It is trained with Adam and binary cross-entropy.

## Run

Open `heart-disease-prediction.ipynb` in Jupyter or VS Code and run the cells from top to bottom:

```bash
pip install numpy pandas scikit-learn tensorflow
```

The notebook expects `heart_disease_data.csv` to be in the same directory.
