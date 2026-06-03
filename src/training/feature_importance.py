import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

df = pd.read_csv("data/raw/feature_dataset.csv")

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

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    random_state=42
)

model.fit(X, y)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = (
    importance
    .sort_values(
        "Importance",
        ascending=False
    )
)

print(
    importance.head(20)
)

plt.figure(figsize=(10, 8))

plt.barh(
    importance["Feature"].head(15),
    importance["Importance"].head(15)
)

plt.tight_layout()

plt.show()