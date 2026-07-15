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

# 2. Sample Loading Logic using Selectbox
@st.cache_data
def load_sample_data():
    try:
        # Load the top 50 samples from the cleaned CSV
        return pd.read_csv('sample_orders.csv').head(50)
    except Exception:
        return pd.DataFrame()

df_samples = load_sample_data()

def update_from_sample():
    selected = st.session_state.sample_selector
    if selected == "Custom Input" or df_samples.empty:
        return
    
    # Extract the index from "Sample 1", "Sample 2", etc.
    idx = int(selected.split(" ")[1]) - 1
    sample = df_samples.iloc[idx]
    
    # Numeric values
    st.session_state.age = int(sample['Delivery_person_Age']) if pd.notna(sample['Delivery_person_Age']) else 30
    st.session_state.rating = float(sample['Delivery_person_Ratings']) if pd.notna(sample['Delivery_person_Ratings']) else 4.5
    st.session_state.distance = float(sample['distance_km']) if pd.notna(sample['distance_km']) else 5.0
    
    # Date and Time
    try:
        st.session_state.order_date = datetime.strptime(str(sample['Order_Date']), '%d-%m-%Y').date()
    except:
        pass 
        
    try:
        st.session_state.order_time = datetime.strptime(str(sample['Time_Orderd']), '%H:%M:%S').time()
    except:
        pass

    # Categoricals
    st.session_state.traffic = str(sample['Road_traffic_density']).strip()
    
    weather = str(sample['Weatherconditions']).replace('conditions ', '').strip()
    if weather == 'Sand': 
        weather = 'Sandstorms'
    if weather in ['Sunny', 'Cloudy', 'Windy', 'Fog', 'Sandstorms', 'Stormy']:
        st.session_state.weather = weather
        
    festival = str(sample['Festival']).strip()
    if festival in ['No', 'Yes']: 
        st.session_state.festival = festival
    
    vehicle = str(sample['Type_of_vehicle']).strip()
    if vehicle in ['motorcycle', 'scooter', 'electric_scooter', 'bicycle']: 
        st.session_state.vehicle = vehicle
    
    order = str(sample['Type_of_order']).strip()
    if order in ['Snack', 'Drinks', 'Buffet', 'Meal']: 
        st.session_state.order = order
    
    city = str(sample['City']).strip()
    if city in ['Urban', 'Metropolitian', 'Semi-Urban']: 
        st.session_state.city = city

# Dropdown for selecting a sample
if not df_samples.empty:
    sample_options = ["Custom Input"] + [f"Sample {i+1}" for i in range(len(df_samples))]
    st.selectbox("Load Data from 50 Pre-configured Samples", options=sample_options, key='sample_selector', on_change=update_from_sample)
    st.markdown("---")

# 3. Initialize Session State defaults so we don't pass 'value' directly to widgets
if 'age' not in st.session_state: st.session_state.age = 30
if 'rating' not in st.session_state: st.session_state.rating = 4.5
if 'distance' not in st.session_state: st.session_state.distance = 5.0
if 'order_date' not in st.session_state: st.session_state.order_date = datetime.today()
if 'order_time' not in st.session_state: st.session_state.order_time = datetime.now().time()

# 4. Collect Inputs (Removed 'value=...' arguments to avoid conflicts)
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Delivery Person Age", min_value=18, max_value=65, key='age')
    rating = st.number_input("Delivery Person Rating", min_value=1.0, max_value=5.0, key='rating')
    distance = st.number_input("Distance (km)", min_value=0.0, max_value=50.0, key='distance')
    
    order_date = st.date_input("Order Date", key='order_date')
    order_time = st.time_input("Order Time", key='order_time')

with col2:
    traffic = st.selectbox("Road Traffic Density", ['Low', 'Medium', 'High', 'Jam'], key='traffic')
    weather = st.selectbox("Weather Conditions", ['Sunny', 'Cloudy', 'Windy', 'Fog', 'Sandstorms', 'Stormy'], key='weather')
    festival = st.selectbox("Is it a Festival?", ['No', 'Yes'], key='festival')
    
    vehicle_type = st.selectbox("Type of Vehicle", ['motorcycle', 'scooter', 'electric_scooter', 'bicycle'], key='vehicle')
    order_type = st.selectbox("Type of Order", ['Snack', 'Drinks', 'Buffet', 'Meal'], key='order')
    city_type = st.selectbox("City Type", ['Urban', 'Metropolitian', 'Semi-Urban'], key='city')

# 5. Process Inputs when User Clicks Predict
if st.button("Predict Delivery Time", type="primary"):
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
        'Type_of_vehicle': vehicle_type,
        'Type_of_order': order_type,
        'City': city_type
    }
    
    user_df = pd.DataFrame([input_dict])

    # Apply One-Hot Encoding
    nominal_cols = ['Type_of_order', 'Type_of_vehicle', 'City']
    user_df = pd.get_dummies(user_df, columns=nominal_cols)

    # Reindex to match the training feature columns exactly
    user_df = user_df.reindex(columns=feature_columns, fill_value=0)

    # Predict
    prediction = model.predict(user_df)[0]
    
    st.success(f"🚚 Estimated Delivery Time: **{prediction:.2f} minutes**")