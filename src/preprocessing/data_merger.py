from pathlib import Path
import pandas as pd


class DataMerger:
    """
    Combines train, stores, and features datasets
    into one master dataset.
    """

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    def load_data(self):
        """
        Load all CSV files.
        """

        train_df = pd.read_csv(self.data_path / "train.csv")
        stores_df = pd.read_csv(self.data_path / "stores.csv")
        features_df = pd.read_csv(self.data_path / "features.csv")

        return train_df, stores_df, features_df

    def merge_data(self):
        """
        Merge datasets together.
        """

        train_df, stores_df, features_df = self.load_data()

        # Merge store information
        merged_df = train_df.merge(
            stores_df,
            on="Store",
            how="left"
        )

        # Merge economic and holiday features
        merged_df = merged_df.merge(
            features_df,
            on=["Store", "Date", "IsHoliday"],
            how="left"
        )

        return merged_df

    def save_data(self, df):
        """
        Save merged dataset.
        """

        output_path = self.data_path / "master_dataset.csv"

        df.to_csv(output_path, index=False)

        print(f"Dataset saved to: {output_path}")


if __name__ == "__main__":

    merger = DataMerger("data/raw")

    master_df = merger.merge_data()

    print("\nMerged Dataset Shape:")
    print(master_df.shape)

    print("\nFirst 5 Rows:")
    print(master_df.head())

    merger.save_data(master_df)