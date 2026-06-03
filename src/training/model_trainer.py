from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error
)


class ModelTrainer:

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    def load_data(self):

        df = pd.read_csv(
            self.data_path / "feature_dataset.csv"
        )

        return df

    def preprocess(self, df):

        print("Original Shape:")
        print(df.shape)

        markdown_cols = [
            "MarkDown1",
            "MarkDown2",
            "MarkDown3",
            "MarkDown4",
            "MarkDown5"
        ]

        df[markdown_cols] = df[markdown_cols].fillna(0)

        lag_cols = [
            "Lag_1",
            "Lag_4",
            "Lag_12",
            "RollingMean_4"
        ]

        df = df.dropna(subset=lag_cols)

        print("\nAfter Cleaning:")
        print(df.shape)

        print("\nAfter Drop NA:")
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

        print("\nTraining Model...")

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

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

        return model


if __name__ == "__main__":

    trainer = ModelTrainer(
        "data/raw"
    )

    trainer.train()