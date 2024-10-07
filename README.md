# MLUno

MLUno is a Python package that offers utilities for data simulation, regression methods, conformal prediction, regression metrics, and plotting. It also provides a Quarto-based website for API documentation and reference.

## Features

- **Data Simulation**: Easily generate synthetic datasets for testing models.
- **Regressors**: Implements KNN and Linear Regression for predictive tasks.
- **Conformal Prediction**: Generates prediction intervals with statistical guarantees.
- **Metrics**: Calculate RMSE, MAE, coverage, and sharpness to evaluate model performance.
- **Plotting**: Visualize data and model predictions with built-in plotting tools.

## Installation

Install MLUno via pip:

```bash
pip install mluno
```
## Usage

- **Data Simulation and Splitting**
  
```python
  from mluno.data import make_line_data, split_data
  
  X, y = make_line_data(n=100)
  X_train, X_test, y_train, y_test = split_data(X, y)
```
  
- **Regression Methods**
  
```python
from mluno.regressors import KNNRegressor

knn = KNNRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
```
- **Conformal Prediction**

```python
from mluno.conformal import ConformalPredictor

conformal = ConformalPredictor(base_regressor=knn)
conformal.fit(X_train, y_train)
intervals = conformal.predict_intervals(X_test)
```
- **Metrics Calculation**

```python
from mluno.metrics import rmse, mae

rmse_value = rmse(y_test, predictions)
mae_value = mae(y_test, predictions)
```

## Documentaion

Explore the full API documentation and examples on the Quarto-generated website!


<!-- # CS 498 End-to-End Data Science MP 02 (sp24) repo for NetID: bsundar3

GitHub username at initialization time: bhv7899

For next steps, please refer to the instructions provided by your course. -->
