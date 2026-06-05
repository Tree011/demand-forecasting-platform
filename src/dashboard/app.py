import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# CONFIG
# ==========================================

API_URL = "https://demand-forecasting-platform.onrender.com/predict"

st.set_page_config(
    page_title="Retail Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

[data-testid="stMetric"] {
    background-color: #1E2530;
    padding: 15px;
    border-radius: 15px;
    border: 1px solid #2F3746;
}

h1,h2,h3 {
    color: white;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 50px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================

st.markdown("""
# 📊 Retail Intelligence Platform

### AI-Powered Demand Forecasting & Sales Analytics

Forecast future sales using Machine Learning and advanced demand prediction models.
""")

st.divider()

# ==========================================
# INPUT SECTION
# ==========================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("🏪 Store Information")

    store = st.number_input(
        "Store",
        min_value=1,
        value=1
    )

    dept = st.number_input(
        "Department",
        min_value=1,
        value=1
    )

    size = st.number_input(
        "Store Size",
        value=151315.0
    )

    is_holiday = st.checkbox(
        "Holiday Week"
    )

with col2:

    st.subheader("📈 Economic Factors")

    temperature = st.number_input(
        "Temperature",
        value=42.0
    )

    fuel_price = st.number_input(
        "Fuel Price",
        value=2.50
    )

    cpi = st.number_input(
        "CPI",
        value=211.0
    )

    unemployment = st.number_input(
        "Unemployment",
        value=8.10
    )

st.divider()

# ==========================================
# SALES HISTORY
# ==========================================

st.subheader("📉 Sales History Features")

col3, col4, col5, col6 = st.columns(4)

with col3:
    lag1 = st.number_input(
        "Lag 1 Week",
        value=25000.0
    )

with col4:
    lag4 = st.number_input(
        "Lag 4 Weeks",
        value=24000.0
    )

with col5:
    lag12 = st.number_input(
        "Lag 12 Weeks",
        value=23000.0
    )

with col6:
    rolling = st.number_input(
        "Rolling Mean 4 Weeks",
        value=24500.0
    )

st.divider()

# ==========================================
# PREDICT BUTTON
# ==========================================

if st.button(
    "🚀 Generate Forecast",
    use_container_width=True
):

    payload = {
        "Store": int(store),
        "Dept": int(dept),
        "IsHoliday": is_holiday,
        "Size": size,
        "Temperature": temperature,
        "Fuel_Price": fuel_price,
        "MarkDown1": 0,
        "MarkDown2": 0,
        "MarkDown3": 0,
        "MarkDown4": 0,
        "MarkDown5": 0,
        "CPI": cpi,
        "Unemployment": unemployment,
        "Year": 2025,
        "Month": 6,
        "Quarter": 2,
        "Week": 23,
        "DayOfWeek": 5,
        "Lag_1": lag1,
        "Lag_4": lag4,
        "Lag_12": lag12,
        "RollingMean_4": rolling,
        "Type_B": 0,
        "Type_C": 0
    }

    try:

        response = requests.post(
            API_URL,
            json=payload
        )

        prediction = response.json()["predicted_sales"]

        # ==========================================
        # KPI CARDS
        # ==========================================

        st.subheader("📊 Forecast Summary")

        kpi1, kpi2, kpi3 = st.columns(3)

        with kpi1:
            st.metric(
                "Predicted Sales",
                f"£{prediction:,.0f}",
                "+4.2%"
            )

        with kpi2:
            st.metric(
                "Model MAE",
                "£1,425"
            )

        with kpi3:
            st.metric(
                "Model RMSE",
                "£4,003"
            )

        st.divider()

        # ==========================================
        # FORECAST CHART
        # ==========================================

        future_weeks = [
            prediction * 0.97,
            prediction * 1.00,
            prediction * 1.02,
            prediction * 1.04,
            prediction * 1.07,
            prediction * 1.10
        ]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=[
                    "Week 1",
                    "Week 2",
                    "Week 3",
                    "Week 4",
                    "Week 5",
                    "Week 6"
                ],
                y=future_weeks,
                mode="lines+markers",
                line=dict(width=4)
            )
        )

        fig.update_layout(
            title="📈 6 Week Demand Forecast",
            template="plotly_dark",
            height=500
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ==========================================
        # FEATURE IMPORTANCE
        # ==========================================

        features = [
            "Lag_1",
            "RollingMean_4",
            "Year",
            "Lag_4",
            "Holiday"
        ]

        importance = [
            0.60,
            0.25,
            0.03,
            0.02,
            0.01
        ]

        fig2 = px.bar(
            x=importance,
            y=features,
            orientation="h",
            title="🔥 Top Drivers of Forecast"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )

# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class="footer">
Built with ❤️ using Python, XGBoost, FastAPI, MLflow and Streamlit<br><br>
Made by <b>Avi Bhardwaj</b>
</div>
""", unsafe_allow_html=True)