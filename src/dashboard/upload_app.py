import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import streamlit as st
import pandas as pd
import plotly.express as px

from feature_engineering.feature_engineering import create_features
from training.trainer import train_forecasting_model
from forecasting.predictor import forecast_future


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="ForecastIQ",
    page_icon="📈",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.block-container{
    padding-top:1rem;
}

[data-testid="stMetric"]{
    background-color:#111827;
    border:1px solid #374151;
    border-radius:12px;
    padding:15px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("📈 ForecastIQ")

    st.markdown("""
### Enterprise Forecasting Platform

Forecast any business metric using machine learning.

---

### Features

✅ Automated Feature Engineering

✅ XGBoost Forecasting

✅ Interactive Visualisations

✅ Forecast Export

✅ Docker Ready

---

### Author

**Avi Bhardwaj**
""")

# =====================================================
# HEADER
# =====================================================

st.markdown("""
# 📈 ForecastIQ

### Enterprise AI Forecasting Platform

Forecast Sales • Revenue • Demand • Inventory • Bookings

Built using Machine Learning & Time Series Engineering

---
""")

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

# =====================================================
# MAIN APP
# =====================================================

if uploaded_file:

    try:

        df = pd.read_csv(uploaded_file)

        st.subheader("📊 Dataset Overview")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Rows",
                f"{df.shape[0]:,}"
            )

        with c2:
            st.metric(
                "Columns",
                df.shape[1]
            )

        with c3:
            st.metric(
                "Missing Values",
                int(df.isna().sum().sum())
            )

        with c4:
            st.metric(
                "Memory MB",
                round(
                    df.memory_usage().sum() / 1024 / 1024,
                    2
                )
            )

        st.dataframe(
            df.head(),
            width="stretch"
        )

        st.divider()

        # =====================================================
        # CONFIGURATION
        # =====================================================

        st.subheader(
            "⚙ Forecast Configuration"
        )

        col1, col2 = st.columns(2)

        with col1:

            date_column = st.selectbox(
                "Date Column",
                df.columns
            )

        with col2:

            target_column = st.selectbox(
                "Target Column",
                df.columns
            )

        feature_columns_selected = st.multiselect(
            "Additional Business Features (Future Enhancement)",
            [
                col
                for col in df.columns
                if col not in [date_column, target_column]
            ]
        )

        forecast_periods = st.slider(
            "Forecast Horizon",
            min_value=1,
            max_value=52,
            value=12
        )

        st.divider()

        # =====================================================
        # TRAIN
        # =====================================================

        if st.button(
            "🚀 Generate Forecast",
            use_container_width=True
        ):

            with st.spinner(
                "Training forecasting model..."
            ):

                engineered_df = create_features(
                    df,
                    date_column,
                    target_column
                )

                (
                    model,
                    mae,
                    rmse,
                    predictions,
                    y_test,
                    feature_columns
                ) = train_forecasting_model(
                    engineered_df,
                    target_column
                )

                forecast_df = forecast_future(
                    model,
                    engineered_df,
                    date_column,
                    target_column,
                    periods=forecast_periods
                )

            st.success(
                "Forecast generated successfully."
            )

            # =====================================================
            # KPI SECTION
            # =====================================================

            k1, k2, k3 = st.columns(3)

            with k1:
                st.metric(
                    "MAE",
                    round(mae, 2)
                )

            with k2:
                st.metric(
                    "RMSE",
                    round(rmse, 2)
                )

            with k3:
                st.metric(
                    "Forecast Horizon",
                    forecast_periods
                )

            st.divider()

            # =====================================================
            # TABS
            # =====================================================

            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Dataset",
                "🧠 Model",
                "🔮 Forecast",
                "⚙ Analytics"
            ])

            # =====================================================
            # DATASET TAB
            # =====================================================

            with tab1:

                st.subheader(
                    "📈 Historical Trend"
                )

                fig_history = px.line(
                    engineered_df,
                    x=date_column,
                    y=target_column,
                    title="Historical Data"
                )

                fig_history.update_layout(
                    template="plotly_dark",
                    height=550
                )

                st.plotly_chart(
                    fig_history,
                    width="stretch"
                )

            # =====================================================
            # MODEL TAB
            # =====================================================

            with tab2:

                st.subheader(
                    "🎯 Actual vs Predicted"
                )

                comparison_df = pd.DataFrame({
                    "Actual": y_test.values,
                    "Predicted": predictions
                })

                fig_actual = px.line(
                    comparison_df,
                    title="Model Performance"
                )

                fig_actual.update_layout(
                    template="plotly_dark",
                    height=550
                )

                st.plotly_chart(
                    fig_actual,
                    width="stretch"
                )

            # =====================================================
            # FORECAST TAB
            # =====================================================

            with tab3:

                st.subheader(
                    "🔮 Future Forecast"
                )

                fig_forecast = px.line(
                    forecast_df,
                    x="Date",
                    y="Forecast",
                    title="Forecast"
                )

                fig_forecast.update_layout(
                    template="plotly_dark",
                    height=550
                )

                st.plotly_chart(
                    fig_forecast,
                    width="stretch"
                )

                st.subheader(
                    "Forecast Results"
                )

                st.dataframe(
                    forecast_df,
                    width="stretch"
                )

                csv = forecast_df.to_csv(
                    index=False
                )

                st.download_button(
                    label="📥 Download Forecast CSV",
                    data=csv,
                    file_name="forecast.csv",
                    mime="text/csv"
                )

            # =====================================================
            # ANALYTICS TAB
            # =====================================================

            with tab4:

                st.subheader(
                    "⚙ Feature Importance"
                )

                importance_df = pd.DataFrame({
                    "Feature": feature_columns,
                    "Importance": model.feature_importances_
                })

                importance_df = (
                    importance_df
                    .sort_values(
                        "Importance",
                        ascending=False
                    )
                )

                fig_importance = px.bar(
                    importance_df,
                    x="Importance",
                    y="Feature",
                    orientation="h"
                )

                fig_importance.update_layout(
                    template="plotly_dark",
                    height=550
                )

                st.plotly_chart(
                    fig_importance,
                    width="stretch"
                )

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown("""
<center>

### ForecastIQ v1.0

Built by Avi Bhardwaj

Machine Learning • Forecasting • Data Science

</center>
""", unsafe_allow_html=True)