import pandas as pd
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1)/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1)/2.0)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

print("Fetching and cleaning test.csv...")
df = pd.read_csv('test.csv')

# Force converting strings to numbers, invalid garbage becomes 'NaN'
cols_to_numeric = ['Delivery_person_Age', 'Delivery_person_Ratings', 
                   'Restaurant_latitude', 'Restaurant_longitude', 
                   'Delivery_location_latitude', 'Delivery_location_longitude']

for col in cols_to_numeric:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Calculating true distance
df['distance_km'] = haversine(df['Restaurant_latitude'], df['Restaurant_longitude'], 
                              df['Delivery_location_latitude'], df['Delivery_location_longitude'])

# Keeping only logical, real-world constraints
df_clean = df[
    (df['Delivery_person_Age'] >= 18) & (df['Delivery_person_Age'] <= 50) &
    (df['Delivery_person_Ratings'] >= 1.0) & (df['Delivery_person_Ratings'] <= 5.0) &
    (df['distance_km'] > 0.5) & (df['distance_km'] <= 50.0)
]

# Generating a representative sample of 200 rows for demo purposes
df_final = df_clean.sample(200, random_state=42) 
df_final.to_csv('sample_orders.csv', index=False)

print("✅ 'sample_orders.csv' generated successfully. Data cleaned and outliers removed.")