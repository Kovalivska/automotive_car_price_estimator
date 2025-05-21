import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os

# --- Load artifacts ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

# Load saved components
model = joblib.load(os.path.join(MODELS_DIR, "xgboost_log_model.pkl"))  # <- Correct XGBoost model
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
label_encoders = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))
X_train_columns = joblib.load(os.path.join(MODELS_DIR, "X_train_columns.pkl"))
segment_results = pd.read_csv(os.path.join(MODELS_DIR, "segment_rmse.csv"))

# --- Prediction Logic ---
def predict_car_price(input_data):
    df = pd.DataFrame([input_data])

    # Feature engineering
    df['vehicle_age'] = 2025 - df['year']
    df['km_per_year'] = df['mileage_km'] / (df['vehicle_age'] + 1)
    df['is_modern_luxury'] = ((df['year'] > 2015) & (df['volume_liters'] > 3.0)).astype(int)
    df['is_low_usage_old'] = ((df['vehicle_age'] > 10) & (df['mileage_km'] < 20000)).astype(int)
    df['age_squared'] = df['vehicle_age'] ** 2

    # Label Encoding
    for col, le in label_encoders.items():
        if df[col].iloc[0] in le.classes_:
            df[col] = le.transform(df[col])
        else:
            df[col] = -1

    # One-hot encoding
    cat_cols = ['condition', 'fuel_type', 'color', 'transmission', 'drive_unit', 'segment']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df = df.reindex(columns=X_train_columns, fill_value=0)

    # Scale numeric features
    df_scaled = scaler.transform(df)

    # Predict (model outputs log(price), need to apply expm1)
    predicted_log_price = model.predict(df_scaled)[0]
    predicted_price = np.expm1(predicted_log_price)

    # Get RMSE for segment
    segment = input_data['segment']
    try:
        segment_rmse = segment_results.loc[segment_results['segment'] == segment, 'RMSE'].values[0]
    except IndexError:
        segment_rmse = segment_results['RMSE'].mean()

    lower_bound = predicted_price - segment_rmse
    upper_bound = predicted_price + segment_rmse
    return round(predicted_price, 2), round(lower_bound, 2), round(upper_bound, 2)

# --- Plotting ---
def plot_price_prediction(price, lower, upper):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lower, upper], y=[1, 1], mode="lines",
                             line=dict(color="lightblue", width=12),
                             name="Confidence Interval"))
    fig.add_trace(go.Scatter(x=[price], y=[1], mode="markers+text",
                             marker=dict(color="darkblue", size=12),
                             text=[f"${price:.0f}"],
                             textposition="top center", name="Predicted Price"))
    fig.update_layout(
        title="Predicted Car Price with Confidence Interval",
        xaxis_title="Price (USD)",
        yaxis=dict(showticklabels=False),
        xaxis=dict(range=[lower - (upper - lower)*0.2, upper + (upper - lower)*0.2]),
        height=300,
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Streamlit UI ---
st.title("🚗 Car Price Prediction App")
st.caption("Model trained on historical car prices (up to 2019) using XGBoost. Predictions are based on data available at that time.")


make_options = sorted(label_encoders['make'].classes_.tolist())
model_options = sorted(label_encoders['model'].classes_.tolist())
make = st.selectbox("Make", make_options, index=make_options.index("bmw") if "bmw" in make_options else 0)
model_input = st.selectbox("Model", model_options, index=model_options.index("x5") if "x5" in model_options else 0)
year = st.number_input("Year", min_value=1950, max_value=2019, value=2018)
mileage_km = st.number_input("Mileage (km)", value=80000)
volume_liters = st.number_input("Engine Volume (L)", value=2.0)
condition = st.selectbox("Condition", ['with mileage', 'with damage', 'for parts'])
fuel_type = st.selectbox("Fuel Type", ['petrol', 'diesel', 'electrocar'])
color = st.selectbox("Color", ['black', 'white', 'gray', 'silver', 'red', 'blue', 'other'])
transmission = st.selectbox("Transmission", ['auto', 'mechanics'])
drive_unit = st.selectbox("Drive Unit", ['front-wheel drive', 'rear drive', 'all-wheel drive', 'part-time four-wheel drive'])
segment = st.selectbox("Segment", ['A', 'B', 'C', 'D', 'E', 'F', 'J', 'M', 'S', 'unknown'])

input_data = {
    'make': make,
    'model': model_input,
    'year': year,
    'mileage_km': mileage_km,
    'volume_liters': volume_liters,
    'condition': condition,
    'fuel_type': fuel_type,
    'color': color,
    'transmission': transmission,
    'drive_unit': drive_unit,
    'segment': segment
}

if st.button("🔮 Predict Price"):
    pred, low, high = predict_car_price(input_data)
    st.success(f"Predicted Price: ${int(pred):,}")
    #st.markdown(f"**Estimated Price Range:**  ${int(low):,} – ${int(high):,}")
   
    plot_price_prediction(pred, low, high)
