import streamlit as st
import joblib
import numpy as np
import pandas as pd
import sklearn
import sklearn.compose._column_transformer as ct

# HOTFIX for custom object used in pipeline
class _RemainderColsList(list): pass
ct._RemainderColsList = _RemainderColsList

st.title("🏠 California House Price Predictor")

# Take user inputs
longitude = st.number_input("Longitude", value=-120.0)
latitude = st.number_input("Latitude", value=37.0)
housing_age = st.number_input("Housing Median Age", value=30)
rooms = st.number_input("Total Rooms", value=2000)
bedrooms = st.number_input("Total Bedrooms", value=400)
population = st.number_input("Population", value=1000)
households = st.number_input("Households", value=500)
income = st.slider("Median Income", 0.0, 15.0, 3.0)
ocean = st.selectbox("Ocean Proximity", ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"])

# 👇 Use DataFrame with correct column names
data = pd.DataFrame({
    "longitude": [longitude],
    "latitude": [latitude],
    "housing_median_age": [housing_age],
    "total_rooms": [rooms],
    "total_bedrooms": [bedrooms],
    "population": [population],
    "households": [households],
    "median_income": [income],
    "ocean_proximity": [ocean]
})

# Load model and predict
model = joblib.load("model/pipeline.pkl")
prediction = model.predict(data)

st.write("💰 Predicted House Price: $", int(prediction[0]))
