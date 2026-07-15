import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import math

# 1. Load the model and the exact feature columns used during training
@st.cache_resource
def load_artifacts():
    model = joblib.load('xgboost_eta_model.pkl')
    feature_columns = joblib.load('logisurge_features.pkl')
    return model, feature_columns

model, feature_columns = load_artifacts()

st.title("LogiSurge: Delivery SLA Predictor")
st.markdown("Predicts the estimated delivery time based on real-time logistics data.")

# 2. Collect Inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Delivery Person Age", min_value=18, max_value=65, value=30)
    rating = st.number_input("Delivery Person Rating", min_value=1.0, max_value=5.0, value=4.5)
    distance = st.number_input("Distance (km)", min_value=0.0, max_value=50.0, value=5.0)
    
    order_date = st.date_input("Order Date", datetime.today())
    order_time = st.time_input("Order Time", datetime.now().time())

with col2:
    traffic = st.selectbox("Road Traffic Density", ['Low', 'Medium', 'High', 'Jam'])
    weather = st.selectbox("Weather Conditions", ['Sunny', 'Cloudy', 'Windy', 'Fog', 'Sandstorms', 'Stormy'])
    festival = st.selectbox("Is it a Festival?", ['No', 'Yes'])
    
    vehicle_type = st.selectbox("Type of Vehicle", ['motorcycle', 'scooter', 'electric_scooter', 'bicycle'])
    order_type = st.selectbox("Type of Order", ['Snack', 'Drinks', 'Buffet', 'Meal'])
    city_type = st.selectbox("City Type", ['Urban', 'Metropolitian', 'Semi-Urban'])

# 3. Process Inputs when User Clicks Predict
if st.button("Predict Delivery Time"):
    # Encoding dictionaries (Must match Kaggle notebook exactly)
    traffic_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Jam': 3}
    weather_map = {'Sunny': 0, 'Cloudy': 1, 'Windy': 2, 'Fog': 3, 'Sandstorms': 4, 'Stormy': 5}
    festival_map = {'No': 0, 'Yes': 1}

    # Time Engineering (Cyclic Encoding)
    decimal_hours = order_time.hour + (order_time.minute / 60.0) + (order_time.second / 3600.0)
    time_sin = np.sin(2 * np.pi * decimal_hours / 24)
    time_cos = np.cos(2 * np.pi * decimal_hours / 24)

    # Date Engineering
    day_of_week = order_date.weekday()
    is_weekend = 1 if day_of_week in [5, 6] else 0

    # Build the input DataFrame
    input_dict = {
        'Delivery_person_Age': age,
        'Delivery_person_Ratings': rating,
        'distance_km': distance,
        'Road_traffic_density': traffic_map[traffic],
        'Weatherconditions': weather_map[weather],
        'Festival': festival_map[festival],
        'Day_of_Week': day_of_week,
        'Is_Weekend': is_weekend,
        'Time_Orderd_sin': time_sin,
        'Time_Orderd_cos': time_cos,
        # Nominal columns for One-Hot Encoding
        'Type_of_vehicle': vehicle_type,
        'Type_of_order': order_type,
        'City': city_type
    }
    
    user_df = pd.DataFrame([input_dict])

    # 4. Apply One-Hot Encoding
    nominal_cols = ['Type_of_order', 'Type_of_vehicle', 'City']
    user_df = pd.get_dummies(user_df, columns=nominal_cols)

    # 5. Reindex to match the training feature columns exactly
    # This prevents the silent dropping bug. Any missing one-hot column gets filled with 0.
    user_df = user_df.reindex(columns=feature_columns, fill_value=0)

    # 6. Predict
    prediction = model.predict(user_df)[0]
    
    st.success(f"Estimated Delivery Time: {prediction:.2f} minutes")