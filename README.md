# ForecastIQ

AI-powered forecasting platform capable of forecasting any business metric from historical time-series data.

## Overview

ForecastIQ enables businesses to upload historical datasets and generate future forecasts using machine learning.

Unlike traditional forecasting tools that require predefined schemas, ForecastIQ allows users to select their own date and target columns, making it adaptable across industries such as retail, hospitality, manufacturing, e-commerce, logistics, and finance.

---

## Features

### Data Upload

* Upload CSV datasets
* Preview uploaded data
* Select custom date columns
* Select custom target columns

### Automated Feature Engineering

Automatically creates:

* Year
* Month
* Quarter
* Week
* Day of Week
* Lag Features
* Rolling Averages

### Machine Learning Forecasting

Uses:

* XGBoost Regressor
* Time-series feature engineering
* Recursive forecasting

### Model Evaluation

Displays:

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* Actual vs Predicted Performance

### Visual Analytics

Interactive Plotly charts:

* Historical Trends
* Future Forecasts
* Feature Importance

### Export Results

* Download forecast results as CSV

---

## Tech Stack

### Frontend

* Streamlit

### Machine Learning

* XGBoost
* Scikit-learn

### Data Processing

* Pandas
* NumPy

### Visualisation

* Plotly

### Deployment

* Docker
* Render

---

## Example Use Cases

### Retail

Forecast:

* Sales
* Revenue
* Transactions

### Hospitality

Forecast:

* Bookings
* Occupancy
* Revenue

### Manufacturing

Forecast:

* Production Volume
* Demand
* Inventory Requirements

### E-Commerce

Forecast:

* Orders
* Revenue
* Customer Activity

---

## Project Structure

src/

├── dashboard/

├── feature_engineering/

├── forecasting/

├── training/

└── api/

---

## Future Roadmap

* Multiple Model Comparison
* LightGBM Support
* Auto Frequency Detection
* Additional Business Features
* Model Persistence
* API Deployment
* MLOps Monitoring

---

## Author

Avi Bhardwaj

ML Engineer
