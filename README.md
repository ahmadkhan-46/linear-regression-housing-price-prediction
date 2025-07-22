# Boston Housing Price Prediction

This project predicts housing prices in Boston using machine learning models.

# Dataset
The Boston Housing dataset contains information about various houses in Boston including:
- CRIM: per capita crime rate by town
- ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
- INDUS: proportion of non-retail business acres per town
- CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- NOX: nitric oxides concentration (parts per 10 million)
- RM: average number of rooms per dwelling
- AGE: proportion of owner-occupied units built prior to 1940
- DIS: weighted distances to five Boston employment centres
- RAD: index of accessibility to radial highways
- TAX: full-value property-tax rate per $10,000
- PTRATIO: pupil-teacher ratio by town
- B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
- LSTAT: % lower status of the population
- PRICE: Median value of owner-occupied homes in $1000's

# Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the preprocessing script: `python scripts/data_preprocessing.py`
4. Train models: `python scripts/train_model.py`
5. Make predictions: `python scripts/predict.py`

# Results

The project implements two models:
 Linear Regression
 Random Forest Regression

Performance metrics are displayed during training.

# Usage

To make predictions, modify the example input in `predict.py` or create a new script that calls the `predict_price()` function with your desired input values.

# Dataset

Dataset is from (https://www.kaggle.com/datasets/vikrishnan/boston-house-prices) because the link you provide is not working or showing error because of some constraints.