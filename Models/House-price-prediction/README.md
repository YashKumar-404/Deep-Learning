# House Price Prediction

A regression model that predicts Boston housing prices from tabular housing features.

## Dataset

`BostonHousing.csv` contains the input features and a `price` column containing the target house price.

## Workflow

The notebook:

1. Loads the dataset and separates features from `price`.
2. Creates an 80/20 train-test split.
3. Standardizes the features with `StandardScaler` fitted on the training data.
4. Trains a TensorFlow/Keras regression network.
5. Evaluates mean squared error and mean absolute error on the training and test sets.

## Model

The network uses dense hidden layers with 64, 32, 16, 8, and 4 ReLU units, followed by a single linear output. It is trained with Adam, mean squared error loss, and mean absolute error as a metric.

## Run

Open `house-price-prediction.ipynb` in Jupyter or VS Code and run the cells from top to bottom:

```bash
pip install numpy pandas scikit-learn tensorflow
```

The notebook expects `BostonHousing.csv` to be in the same directory.
