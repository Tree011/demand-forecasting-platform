import pandas as pd


def inspect_csv(file_path):
    df = pd.read_csv(file_path)

    print("\n" + "=" * 50)
    print(f"FILE: {file_path}")
    print("=" * 50)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:") 
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nFirst 5 Rows:")
    print(df.head())


if __name__ == "__main__":

    files = [
        "data/raw/train.csv",
        "data/raw/stores.csv",
        "data/raw/features.csv"
    ]

    for file in files:
        inspect_csv(file)