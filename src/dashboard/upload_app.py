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
# HEADER
# =====================================================

st.title("📈 ForecastIQ")

st.markdown("""
### AI-Powered Forecasting Platform

Upload any business dataset and generate future forecasts.

**Made by Avi Bhardwaj**
""")

st.divider()

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

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head()
        )

        st.write(
            f"Rows: {df.shape[0]} | Columns: {df.shape[1]}"
        )

        st.divider()

        # =====================================================
        # COLUMN SELECTION
        # =====================================================

        col1, col2 = st.columns(2)

        with col1:

            date_column = st.selectbox(
                "Select Date Column",
                df.columns
            )

        with col2:

            target_column = st.selectbox(
                "Select Target Column",
                df.columns
            )

        forecast_periods = st.slider(
            "Forecast Periods",
            min_value=1,
            max_value=52,
            value=12
        )

        st.divider()

        # =====================================================
        # TRAIN MODEL
        # =====================================================

        if st.button("🚀 Generate Forecast"):

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

            st.success(
                "Forecast generated successfully."
            )

            # =====================================================
            # METRICS
            # =====================================================

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "MAE",
                    round(mae, 2)
                )

            with col2:

                st.metric(
                    "RMSE",
                    round(rmse, 2)
                )

            st.divider()

            # =====================================================
            # ACTUAL VS PREDICTED
            # =====================================================

            st.subheader(
                "Actual vs Predicted"
            )

            comparison_df = pd.DataFrame({
                "Actual": y_test.values,
                "Predicted": predictions
            })

            fig_actual = px.line(
                comparison_df,
                title="Model Performance"
            )

            st.plotly_chart(
                fig_actual,
                use_container_width=True
            )

            # =====================================================
            # HISTORICAL DATA
            # =====================================================

            st.subheader(
                "Historical Trend"
            )

            fig_history = px.line(
                engineered_df,
                x=date_column,
                y=target_column,
                title="Historical Data"
            )

            st.plotly_chart(
                fig_history,
                use_container_width=True
            )

            # =====================================================
            # FUTURE FORECAST
            # =====================================================

            forecast_df = forecast_future(
                model,
                engineered_df,
                date_column,
                target_column,
                periods=forecast_periods
            )

            st.subheader(
                "Future Forecast"
            )

            fig_forecast = px.line(
                forecast_df,
                x="Date",
                y="Forecast",
                title="Future Forecast"
            )

            st.plotly_chart(
                fig_forecast,
                use_container_width=True
            )

            # =====================================================
            # FEATURE IMPORTANCE
            # =====================================================

            st.subheader(
                "Feature Importance"
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

            st.plotly_chart(
                fig_importance,
                use_container_width=True
            )

            # =====================================================
            # FORECAST TABLE
            # =====================================================

            st.subheader(
                "Forecast Results"
            )

            st.dataframe(
                forecast_df
            )

            # =====================================================
            # DOWNLOAD
            # =====================================================

            csv = forecast_df.to_csv(
                index=False
            )

            st.download_button(
                label="📥 Download Forecast CSV",
                data=csv,
                file_name="forecast.csv",
                mime="text/csv"
            )

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )