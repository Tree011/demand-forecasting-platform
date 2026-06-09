# ForecastIQ

Enterprise AI Forecasting Platform built using Machine Learning, Time Series Feature Engineering, XGBoost, Streamlit, Docker and Render.

---

## Overview

ForecastIQ is an end-to-end forecasting platform that enables users to upload historical business datasets, automatically engineer forecasting features, train machine learning models and generate future forecasts through an interactive dashboard.

The platform is designed to support a wide range of business forecasting use cases including:

* Sales Forecasting
* Revenue Forecasting
* Demand Forecasting
* Inventory Forecasting
* Booking Forecasting
* Operational Forecasting

---

## Key Features

### Automated Feature Engineering

Automatically creates forecasting features including:

* Year
* Month
* Quarter
* Week
* Day Of Week
* Lag Features
* Rolling Average Features

### Machine Learning Forecasting

* XGBoost Regressor
* Time Series Forecasting
* Multi-Step Forecast Generation
* Recursive Forecasting

### Interactive Dashboard

* CSV Upload
* Dataset Exploration
* Forecast Configuration
* Historical Trend Analysis
* Model Evaluation
* Feature Importance Analysis

### Export Functionality

* Forecast Download
* CSV Export

### Deployment Ready

* Docker Containerisation
* Render Deployment
* GitHub Version Control

---

## Technology Stack

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost

### Dashboard

* Streamlit
* Plotly

### Deployment

* Docker
* Render

### Version Control

* Git
* GitHub

---

## Application Architecture

```text
User Uploads Dataset
           │
           ▼
Feature Engineering
           │
           ▼
Model Training
           │
           ▼
Forecast Generation
           │
           ▼
Interactive Dashboard
           │
           ▼
Forecast Export
```

---

## Model Evaluation

ForecastIQ evaluates forecasting performance using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

These metrics help assess prediction quality and forecasting accuracy.

---

## Screenshots

### Application Overview

![Overview](screenshots/Screenshot%202026-06-09%20at%2011.12.30%E2%80%AFPM.png)

### Dataset Overview

![Dataset](screenshots/Screenshot%202026-06-09%20at%2011.12.41%E2%80%AFPM.png)

### Forecast Configuration

![Configuration](screenshots/Screenshot%202026-06-09%20at%2011.14.43%E2%80%AFPM.png)

### Historical Trend Analysis

![Historical](screenshots/Screenshot%202026-06-09%20at%2011.15.11%E2%80%AFPM.png)

### Forecast Results

![Forecast](screenshots/Screenshot%202026-06-09%20at%2011.22.10%E2%80%AFPM.png)

---

## Example Dataset

The platform has been tested using the Walmart Sales Forecasting dataset containing:

* 421,570 Rows
* Historical Weekly Sales
* Multiple Store Locations
* Holiday Indicators

---

## Local Installation

Clone the repository:

```bash
git clone https://github.com/Tree011/demand-forecasting-platform.git
```

Move into the project directory:

```bash
cd demand-forecasting-platform
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run src/dashboard/upload_app.py
```

---

## Docker

Build the Docker image:

```bash
docker build -t forecastiq .
```

Run the container:

```bash
docker run -p 8501:8501 forecastiq
```

---

## Deployment

The application is containerised using Docker and deployed using Render.

---

## Future Improvements

* Prophet Forecasting
* LSTM Forecasting
* Hyperparameter Optimisation
* Automated Model Selection
* Forecast Comparison Dashboard
* REST API Integration
* Authentication & User Management
* Cloud Storage Integration

---

## Repository Structure

```text
demand-forecasting-platform/
│
├── src/
│   ├── dashboard/
│   ├── forecasting/
│   ├── feature_engineering/
│   └── training/
│
├── screenshots/
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Author

### Avi Bhardwaj

Data Analyst | Machine Learning Engineer | Advanced Computer Science Graduate

---

## License

This project is available for educational, portfolio and demonstration purposes.
