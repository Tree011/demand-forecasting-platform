from pathlib import Path
import pandas as pd


class FeatureEngineer:

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    def load_data(self):
        """
        Load master dataset.
        """

        df = pd.read_csv(
            self.data_path / "master_dataset.csv"
        )

        return df

    def create_date_features(self, df):
        """
        Create features from date column.
        """

        df["Date"] = pd.to_datetime(df["Date"])

        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Quarter"] = df["Date"].dt.quarter
        df["Week"] = df["Date"].dt.isocalendar().week
        df["DayOfWeek"] = df["Date"].dt.dayofweek

        return df

    def create_lag_features(self, df):
        """
        Create previous sales features.
        """

        df = df.sort_values(
            by=["Store", "Dept", "Date"]
        )

        df["Lag_1"] = (
            df.groupby(["Store", "Dept"])["Weekly_Sales"]
            .shift(1)
        )

        df["Lag_4"] = (
            df.groupby(["Store", "Dept"])["Weekly_Sales"]
            .shift(4)
        )

        df["Lag_12"] = (
            df.groupby(["Store", "Dept"])["Weekly_Sales"]
            .shift(12)
        )

        return df

    def create_rolling_features(self, df):
        """
        Rolling average sales.
        """

        df["RollingMean_4"] = (
            df.groupby(["Store", "Dept"])["Weekly_Sales"]
            .transform(
                lambda x:
                x.shift(1).rolling(4).mean()
            )
        )

        return df

    def save_data(self, df):

        output_path = (
            self.data_path /
            "feature_dataset.csv"
        )

        df.to_csv(output_path, index=False)

        print(f"\nSaved: {output_path}")

    def run(self):

        df = self.load_data()

        print("Original Shape:")
        print(df.shape)

        df = self.create_date_features(df)

        df = self.create_lag_features(df)

        df = self.create_rolling_features(df)

        print("\nFinal Shape:")
        print(df.shape)

        print("\nColumns Added:")
        print([
            "Year",
            "Month",
            "Quarter",
            "Week",
            "DayOfWeek",
            "Lag_1",
            "Lag_4",
            "Lag_12",
            "RollingMean_4"
        ])

        self.save_data(df)


if __name__ == "__main__":

    engineer = FeatureEngineer(
        "data/raw"
    )

    engineer.run()