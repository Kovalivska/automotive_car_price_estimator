import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Try to import XGBoost types
try:
    import xgboost as xgb
    BoosterType = xgb.Booster
except ImportError:
    BoosterType = tuple()

# --- Paths ---
HERE        = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.abspath(os.path.join(HERE, os.pardir))
DATA_PATH   = os.path.join(ROOT, "data", "cars.csv")
MODELS_PATH = os.path.join(ROOT, "models")

# --- Load raw data for dropdowns ---
@st.cache_data
def load_raw():
    return pd.read_csv(DATA_PATH)

raw_df = load_raw()

# --- Load artifacts ---
@st.cache_resource
def load_art(path):
    if not os.path.exists(path):
        st.error(f"Missing file: {path}")
        st.stop()
    return joblib.load(path)

model           = load_art(os.path.join(MODELS_PATH, "xgboost_log_model.pkl"))
scaler          = load_art(os.path.join(MODELS_PATH, "scaler.pkl"))
X_train_columns = load_art(os.path.join(MODELS_PATH, "X_train_columns.pkl"))
segment_results = pd.read_csv(os.path.join(MODELS_PATH, "segment_rmse.csv"))

# --- Build dropdowns ---
make_options = sorted(raw_df["make"].unique())

# --- Prediction logic ---
def predict_car_price(inp: dict):
    df = pd.DataFrame([inp])
    # feature engineering
    df["vehicle_age"]      = 2025 - df["year"]
    df["km_per_year"]      = df["mileage_km"] / (df["vehicle_age"] + 1)
    df["is_modern_luxury"] = ((df["year"] > 2015) & (df["volume_liters"] > 3.0)).astype(int)
    df["is_low_usage_old"] = ((df["vehicle_age"] > 10) & (df["mileage_km"] < 20000)).astype(int)
    df["age_squared"]      = df["vehicle_age"] ** 2

    # one-hot encode
    cat_cols = [
      "make","model","condition","fuel_type",
      "color","transmission","drive_unit","segment"
    ]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df = df.reindex(columns=X_train_columns, fill_value=0)

    # scale
    Xs = scaler.transform(df)

    # predict
    if isinstance(model, BoosterType):
        # raw Booster
        dmat = xgb.DMatrix(Xs)
        log_pred = model.predict(dmat)[0]
    else:
        # sklearn Pipeline or XGBRegressor
        log_pred = model.predict(Xs)[0]

    price = np.expm1(log_pred)

    # compute interval
    seg = inp["segment"]
    try:
        rmse = float(segment_results.loc[segment_results["segment"] == seg, "RMSE"])
    except:
        rmse = segment_results["RMSE"].mean()
    low, high = price - rmse, price + rmse

    return round(price,2), round(low,2), round(high,2)

# --- Plot helper ---
def plot_price_prediction(price, low, high):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[low, high], y=[1,1], mode="lines", line=dict(width=12),
        name="Confidence Interval"))
    fig.add_trace(go.Scatter(
        x=[price], y=[1], mode="markers+text", marker=dict(size=12),
        text=[f"${price:,.0f}"], textposition="top center",
        name="Predicted Price"))
    fig.update_layout(
        title="Predicted Car Price",
        xaxis_title="USD",
        yaxis=dict(showticklabels=False),
        xaxis=dict(range=[low - (high-low)*0.2, high + (high-low)*0.2]),
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)


# --- Streamlit UI ---
st.title("🚗 Car Price Prediction")
st.caption("XGBoost model trained on car data up to 2019.")

make        = st.selectbox("Make", make_options)
model_opts  = sorted(raw_df[raw_df["make"] == make]["model"].unique())
model_in    = st.selectbox("Model", model_opts)

year        = st.number_input("Year", 1950, 2019, 2018)
mileage_km  = st.number_input("Mileage (km)", 0, 300_000, 80_000)
volume_l    = st.number_input("Engine Volume (L)", 0.0, 10.0, 2.0, format="%.1f")
condition   = st.selectbox("Condition", ["with mileage","with damage","for parts"])
fuel_type   = st.selectbox("Fuel Type", ["petrol","diesel","electrocar"])
color       = st.selectbox("Color", ["black","white","gray","silver","red","blue","other"])
trans       = st.selectbox("Transmission", ["auto","mechanics"])
drive       = st.selectbox("Drive Unit", [
               "front-wheel drive","rear drive",
               "all-wheel drive","part-time four-wheel drive"])
segment     = st.selectbox("Segment", ["A","B","C","D","E","F","J","M","S","unknown"])

inp = {
    "make": make, "model": model_in, "year": year,
    "mileage_km": mileage_km, "volume_liters": volume_l,
    "condition": condition, "fuel_type": fuel_type,
    "color": color, "transmission": trans,
    "drive_unit": drive, "segment": segment
}

if st.button("🔮 Predict Price"):
    price, low, high = predict_car_price(inp)
    st.success(f"Predicted Price: ${price:,.0f}")
    plot_price_prediction(price, low, high)
