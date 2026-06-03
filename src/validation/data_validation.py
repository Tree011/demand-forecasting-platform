from pathlib import Path
import pandas as pd


class DataValidator:

    REQUIRED_FILES = [
        "train.csv",
        "stores.csv",
        "features.csv"
    ]

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    def validate_files_exist(self):

        print("Checking required files...")

        for file_name in self.REQUIRED_FILES:

            file_path = self.data_path / file_name

            if not file_path.exists():
                raise FileNotFoundError(
                    f"Missing file: {file_name}"
                )

            print(f"✓ {file_name}")

        print("\nAll required files found.")


if __name__ == "__main__":

    validator = DataValidator("data/raw")

    validator.validate_files_exist()