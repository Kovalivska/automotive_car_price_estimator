
---

# 🚗 Car Price Prediction App

This project aims to build a machine learning model that predicts the market price of used cars based on key features such as make, model, year, mileage, fuel type, and more. It also includes a Streamlit web application for interactive prediction.

---

## 🎯 Project Goals

- Estimate used car prices accurately
- Provide a tool for dealerships and private sellers to evaluate car values
- Compare various machine learning models to select the best one
- Package everything into an intuitive Streamlit interface

---

## 🖼️ App Preview

https://kovalivska-automotive-car-price-estimator-scriptsapp-xuiguq.streamlit.app/ 

![Screenshot of the App](/assets/Screenshot_app.png)

---

## 📁 Project Structure

```
automotive_car_price_estimator/
├── assets/                   ← Images, logos, screenshots
│   └── Screenshot_app.png
├── data/                     ← (Optional) Raw or cleaned datasets
├── models/                   ← Saved models and preprocessing
│   ├── best_model.pkl         # Final XGBoost model with log1p target
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── X_train_columns.pkl
│   └── segment_rmse.csv
├── notebooks/                ← Exploratory analysis, model training
│   ├── CAR_PRICE_PREDICTION_MODEL.ipynb
│   └── PyGWalker_for_cars.ipynb
├── reports/                  ← EDA results and reports
│   └── car_pricing_eda.html
├── scripts/                  ← Streamlit application
│   └── app.py
└── requirements.txt          ← Required packages for the project
```

---

## 🤖 Models Tested

The following regression models were trained and evaluated:
- 🔹 LinearRegression 
- 🔹 DecisionTreeRegressor 
- 🔹 GradientBoostingRegressor 
- 🔹 XGBoostRegressor  
- 🔹 LightGBMRegressor  
- 🔹 CatBoostRegressor 

### 📊 Metrics Used

| Model       		| MAE       | RMSE     |
|-------------------|-----------|----------|
| CatBoost    		| 1758.17   | 2968.08  |
| LightGBM    		| 1670.52   | 3039.68  |
| XGBoost     		| 1707.54   | 2935.52  |
| Linear Regression | 2077.56   | 4125.57  |
| Decision Tree 	| 2500.32   | 4419.97  |
| Gradient Boosting | 1725.61   | 3104.63  |

✅ **XGBoost (log-transformed target + tuning) was selected as the final model** with the lowest MAE **$1525.87** and RMSE **$2829.23**, aiming for the most consistent pricing across economy and mid-range cars.

---

## 🚀 How to Run the App

### 1. Clone the project:
```bash
git clone https://github.com/yourusername/automotive_car_price_estimator.git
cd automotive_car_price_estimator
```

### 2. Create a virtual environment (optional):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies:
```bash
pip install -r requirements.txt
```

### 4. Launch Streamlit app:
```bash
cd scripts/
streamlit run app.py
```

Open your browser at: http://localhost:8501

---

## 📦 Main Packages

- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`
- `catboost`
- `lightgbm`
- `xgboost`
- `plotly`
- `matplotlib`
- `seaborn`

Install with:
```bash
pip install -r requirements.txt
```

---

## 🧠 Author

Developed by **Svitlana Kovalivska, PhD, 2025**.  
For questions, feel free to open an issue or reach out on GitHub.

---
