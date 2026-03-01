import streamlit as st
import pandas as pd
import numpy as np
import joblib

rf_model = joblib.load('crop_yield_rf_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

st.set_page_config(page_title="Crop Yield Predictor", layout="wide")

st.title("🌾 Crop Yield Predictor")
st.write("Enter the agricultural and climate details below to estimate crop yield (hg/ha).")

col1, col2 = st.columns(2)

area_options = list(label_encoders['Area'].classes_)
item_options = list(label_encoders['Item'].classes_)

with col1:
    area = st.selectbox("Area (Country)", area_options)
    item = st.selectbox("Item (Crop Type)", item_options)
    year = st.number_input("Year", min_value=1990, max_value=2050, value=2013, step=1)

with col2:
    rainfall = st.number_input("Average Rainfall (mm/year)", min_value=0.0, value=1000.0, step=10.0)
    pesticides = st.number_input("Pesticides (tonnes)", min_value=0.0, value=100.0, step=1.0)
    temp = st.number_input("Average Temperature (°C)", min_value=-10.0, max_value=50.0, value=20.0, step=0.1)

if st.button("Predict Yield", use_container_width=True):
    try:
        area_encoded = label_encoders['Area'].transform([area])[0]
        item_encoded = label_encoders['Item'].transform([item])[0]
        
        input_data = pd.DataFrame({
            'Area': [area_encoded],
            'Item': [item_encoded],
            'Year': [year],
            'average_rain_fall_mm_per_year': [rainfall],
            'pesticides_tonnes': [pesticides],
            'avg_temp': [temp]
        })
        
        numeric_cols = ['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Year']
        input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])
        
        predicted_yield = rf_model.predict(input_data)[0]
        
        st.success(f"### Predicted Yield: {predicted_yield:,.2f} hg/ha")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
