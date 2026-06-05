import pandas as pd


def forecast_future(
    model,
    df,
    date_col,
    target_col,
    periods=12
):

    current_df = df.copy()

    predictions = []

    for _ in range(periods):

        last_date = current_df[
            date_col
        ].max()

        next_date = (
            last_date +
            pd.Timedelta(days=1)
        )

        lag_1 = current_df[
            target_col
        ].iloc[-1]

        if len(current_df) >= 7:
            lag_7 = current_df[
                target_col
            ].iloc[-7]
        else:
            lag_7 = lag_1

        if len(current_df) >= 30:
            lag_30 = current_df[
                target_col
            ].iloc[-30]
        else:
            lag_30 = lag_1

        rolling_mean_7 = (
            current_df[target_col]
            .tail(7)
            .mean()
        )

        rolling_mean_30 = (
            current_df[target_col]
            .tail(30)
            .mean()
        )

        future_features = pd.DataFrame({
            "Year": [next_date.year],
            "Month": [next_date.month],
            "Quarter": [
                (next_date.month - 1) // 3 + 1
            ],
            "Week": [
                next_date.isocalendar().week
            ],
            "DayOfWeek": [
                next_date.dayofweek
            ],
            "Lag_1": [lag_1],
            "Lag_7": [lag_7],
            "Lag_30": [lag_30],
            "RollingMean_7": [
                rolling_mean_7
            ],
            "RollingMean_30": [
                rolling_mean_30
            ]
        })

        forecast_value = (
            model.predict(
                future_features
            )[0]
        )

        predictions.append({
            "Date": next_date,
            "Forecast": forecast_value
        })

        new_row = pd.DataFrame({
            date_col: [next_date],
            target_col: [forecast_value]
        })

        current_df = pd.concat(
            [
                current_df,
                new_row
            ],
            ignore_index=True
        )

    return pd.DataFrame(
        predictions
    )