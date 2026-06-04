from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error
)

from xgboost import XGBRegressor

import mlflow
import mlflow.sklearn


class XGBoostMLflowTrainer:

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    def load_data(self):

        df = pd.read_csv(
            self.data_path / "feature_dataset.csv"
        )

        return df

    def preprocess(self, df):

        markdown_cols = [
            "MarkDown1",
            "MarkDown2",
            "MarkDown3",
            "MarkDown4",
            "MarkDown5"
        ]

        df[markdown_cols] = (
            df[markdown_cols]
            .fillna(0)
        )

        lag_cols = [
            "Lag_1",
            "Lag_4",
            "Lag_12",
            "RollingMean_4"
        ]

        df = df.dropna(
            subset=lag_cols
        )

        print("Shape After Cleaning:")
        print(df.shape)

        return df

    def prepare_features(self, df):

        y = df["Weekly_Sales"]

        X = df.drop(
            columns=[
                "Weekly_Sales",
                "Date"
            ]
        )

        X = pd.get_dummies(
            X,
            columns=["Type"],
            drop_first=True
        )

        return X, y

    def train(self):

        df = self.load_data()

        df = self.preprocess(df)

        X, y = self.prepare_features(df)

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )
        )

        mlflow.set_experiment(
            "Demand Forecasting"
        )

        with mlflow.start_run():

            model = XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            mae = mean_absolute_error(
                y_test,
                predictions
            )

            rmse = root_mean_squared_error(
                y_test,
                predictions
            )

            print("\nResults")
            print("-" * 30)
            print(f"MAE  : {mae:.2f}")
            print(f"RMSE : {rmse:.2f}")

            # Log Parameters
            mlflow.log_param(
                "n_estimators",
                300
            )

            mlflow.log_param(
                "learning_rate",
                0.05
            )

            mlflow.log_param(
                "max_depth",
                8
            )

            # Log Metrics
            mlflow.log_metric(
                "MAE",
                mae
            )

            mlflow.log_metric(
                "RMSE",
                rmse
            )

            # Log Model
            mlflow.sklearn.log_model(
                model,
                "xgboost_model"
            )

            # Save Local Model
            Path(
                "models"
            ).mkdir(
                exist_ok=True
            )

            joblib.dump(
                model,
                "models/xgboost_model.pkl"
            )

            print(
                "\nModel Saved Successfully"
            )

        return model


if __name__ == "__main__":

    trainer = XGBoostMLflowTrainer(
        "data/raw"
    )

    trainer.train()