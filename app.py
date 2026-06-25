import streamlit as st
import joblib
import pandas as pd
import numpy as np


model = joblib.load('xgboost_eta_model.pkl')
features = joblib.load('logisurge_features.pkl')
df_samples = pd.read_csv('sample_orders.csv')

st.set_page_config(layout="wide")
st.title("🚀 LogiSurge: Professional Dispatch Console")


if "age_key" not in st.session_state: st.session_state["age_key"] = 25.0
if "rating_key" not in st.session_state: st.session_state["rating_key"] = 4.0
if "dist_key" not in st.session_state: st.session_state["dist_key"] = 5.0
if "traffic_key" not in st.session_state: st.session_state["traffic_key"] = 0
if "weather_key" not in st.session_state: st.session_state["weather_key"] = 0
if "vehicle_key" not in st.session_state: st.session_state["vehicle_key"] = 0

# Haversine Distance Calculator
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1)/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1)/2.0)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

# Sidebar
sla_limit = st.sidebar.slider("Set SLA Threshold (mins)", 20, 60, 40)
option = st.sidebar.radio("Choose Input:", ["Manual Entry", "Load from Sample"])

# Form Data Handling
if option == "Load from Sample":
    idx = st.sidebar.selectbox("Select Order:", df_samples.index)
    if st.sidebar.button("Fill Values"):
        row = df_samples.loc[idx]
        
        st.session_state["age_key"] = float(row.get('Delivery_person_Age', 25.0))
        st.session_state["rating_key"] = float(row.get('Delivery_person_Ratings', 4.0))
        
        # DISTANCE CALCULATION LOGIC 
        if 'distance_km' in row and not pd.isna(row['distance_km']):
            st.session_state["dist_key"] = float(row['distance_km'])
        else:
            try:
                
                lat1 = float(row['Restaurant_latitude'])
                lon1 = float(row['Restaurant_longitude'])
                lat2 = float(row['Delivery_location_latitude'])
                lon2 = float(row['Delivery_location_longitude'])
                st.session_state["dist_key"] = round(haversine(lat1, lon1, lat2, lon2), 2)
            except KeyError:
                st.session_state["dist_key"] = 5.0 # Absolute fallback
        
        t_str = str(row.get('Road_traffic_density', '')).strip().title()
        traffic_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Jam': 3, 'Med': 1}
        st.session_state["traffic_key"] = traffic_map.get(t_str, 0)
        
        w_str = str(row.get('Weatherconditions', '')).strip().replace('conditions ', '').title()
        weather_map = {'Sunny': 0, 'Cloudy': 1, 'Windy': 2, 'Fog': 3, 'Sand': 4, 'Stormy': 5}
        st.session_state["weather_key"] = weather_map.get(w_str, 0)
        
        v_str = str(row.get('Type_of_vehicle', '')).strip().lower()
        vehicle_map = {'motorcycle': 0, 'scooter': 1, 'electric_scooter': 2}
        st.session_state["vehicle_key"] = vehicle_map.get(v_str, 0)
        
        st.rerun()

# Form Layout Bound to Keys
with st.form("dispatch_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Delivery Person Age", key="age_key")
        rating = st.number_input("Delivery Person Ratings", key="rating_key")
    with col2:
        dist = st.number_input("Distance (km)", key="dist_key")
        traffic = st.selectbox("Road Traffic Density", [0, 1, 2, 3], format_func=lambda x: ['Low', 'Med', 'High', 'Jam'][x], key="traffic_key")
    with col3:
        weather = st.selectbox("Weather Conditions", [0, 1, 2, 3, 4, 5], format_func=lambda x: ['Sunny', 'Cloudy', 'Windy', 'Fog', 'Sand', 'Stormy'][x], key="weather_key")
        vehicle_type = st.selectbox("Type of Vehicle", [0, 1, 2], format_func=lambda x: ['Motorcycle', 'Scooter', 'Electric'][x], key="vehicle_key")

    submitted = st.form_submit_button("Predict SLA Risk")

# Prediction Logic
if submitted:
    input_dict = {f: 0 for f in features} 
    input_dict.update({
        'Delivery_person_Age': age, 
        'Delivery_person_Ratings': rating, 
        'distance_km': dist, 
        'Road_traffic_density': traffic, 
        'Weatherconditions': weather,
        'Type_of_vehicle': vehicle_type
    })
    
    input_df = pd.DataFrame([input_dict]).reindex(columns=features, fill_value=0)
    prediction = model.predict(input_df)[0]
    
    st.write(f"### 🎯 Predicted ETA: {prediction:.1f} minutes")
    if prediction > sla_limit: 
        st.error(f"🚨 CRITICAL: SLA BREACH IMMINENT! (Limit: {sla_limit}m)")
    else: 
        st.success("✅ SLA SAFE: ON-TIME DELIVERY")