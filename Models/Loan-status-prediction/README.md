# Loan Status Prediction

A binary classification model that predicts loan approval status from applicant and loan information.

## Dataset

- `credit_train.csv` contains the training data.
- `credit_test.csv` is available for additional prediction workflows.

The notebook uses `Loan Status` as the target and removes identifier columns such as `Loan ID` and `Customer ID`.

## Workflow

The notebook:

1. Loads and inspects the credit data.
2. Removes identifiers and duplicate records.
3. Drops the highly incomplete `Months since last delinquent` field.
4. Fills selected missing values and removes remaining incomplete rows.
5. Encodes categorical columns with `LabelEncoder`.
6. Creates a stratified train-test split and scales numeric features.
7. Trains a TensorFlow/Keras classifier with dropout and early stopping.
8. Reports accuracy and AUC, then compares results with logistic regression metrics.

## Model

The neural network uses dense layers with 64 and 32 ReLU units, dropout regularization, and a sigmoid output. It is trained with Adam, binary cross-entropy, accuracy, and AUC.

## Run

Open `loan-prediction.ipynb` in Jupyter or VS Code and run the cells from top to bottom:

```bash
pip install numpy pandas seaborn missingno scikit-learn tensorflow
```

The notebook expects `credit_train.csv` and `credit_test.csv` to be in the same directory.
