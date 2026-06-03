from pathlib import Path
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Demand Forecasting API",
    version="1.0"
)

MODEL_PATH = Path("models/xgboost_model.pkl")

model = joblib.load(MODEL_PATH)


class PredictionRequest(BaseModel):
    Store: int
    Dept: int
    IsHoliday: bool
    Size: float
    Temperature: float
    Fuel_Price: float
    MarkDown1: float
    MarkDown2: float
    MarkDown3: float
    MarkDown4: float
    MarkDown5: float
    CPI: float
    Unemployment: float
    Year: int
    Month: int
    Quarter: int
    Week: int
    DayOfWeek: int
    Lag_1: float
    Lag_4: float
    Lag_12: float
    RollingMean_4: float
    Type_B: int = 0
    Type_C: int = 0


@app.get("/")
def home():
    return {
        "message": "Demand Forecasting API Running"
    }


@app.post("/predict")
def predict(data: PredictionRequest):

    df = pd.DataFrame([data.model_dump()])

    prediction = model.predict(df)[0]

    return {
        "predicted_sales": round(float(prediction), 2)
    }