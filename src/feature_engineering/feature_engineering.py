import pandas as pd


def create_features(
    df,
    date_column,
    target_column
):

    df = df.copy()

    # Convert selected date column
    df[date_column] = pd.to_datetime(
        df[date_column],
        errors="coerce"
    )

    # Convert selected target column
    df[target_column] = pd.to_numeric(
        df[target_column],
        errors="coerce"
    )

    # Remove invalid rows
    df = df.dropna(
        subset=[
            date_column,
            target_column
        ]
    )

    # Sort chronologically
    df = df.sort_values(
        date_column
    )

    # ======================
    # Time Features
    # ======================

    df["Year"] = df[date_column].apply(
        lambda x: x.year
    )

    df["Month"] = df[date_column].apply(
        lambda x: x.month
    )

    df["Quarter"] = df[date_column].apply(
        lambda x: x.quarter
    )

    df["Week"] = df[date_column].apply(
        lambda x: x.isocalendar().week
    )

    df["DayOfWeek"] = df[date_column].apply(
        lambda x: x.dayofweek
    )

    # ======================
    # Lag Features
    # ======================

    df["Lag_1"] = (
        df[target_column].shift(1)
    )

    df["Lag_7"] = (
        df[target_column].shift(7)
    )

    df["Lag_30"] = (
        df[target_column].shift(30)
    )

    # ======================
    # Rolling Features
    # ======================

    df["RollingMean_7"] = (
        df[target_column]
        .rolling(7)
        .mean()
    )

    df["RollingMean_30"] = (
        df[target_column]
        .rolling(30)
        .mean()
    )

    df = df.dropna()

    return df