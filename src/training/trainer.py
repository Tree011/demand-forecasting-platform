from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor


def train_forecasting_model(
    df,
    target_column
):

    feature_columns = [
        "Year",
        "Month",
        "Quarter",
        "Week",
        "DayOfWeek",
        "Lag_1",
        "Lag_7",
        "Lag_30",
        "RollingMean_7",
        "RollingMean_30"
    ]

    X = df[feature_columns]

    y = df[target_column]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            shuffle=False
        )
    )

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
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

    rmse = (
        mean_squared_error(
            y_test,
            predictions
        ) ** 0.5
    )

    return (
        model,
        mae,
        rmse,
        predictions,
        y_test,
        feature_columns
    )