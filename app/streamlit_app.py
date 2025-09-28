# streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import requests
import io

st.set_page_config(page_title="🏠 House Price Predictor", layout="centered")

# ---------- Helper: Load from GitHub Releases ----------
def download_from_github_release(url):
    response = requests.get(url)
    response.raise_for_status()
    return pickle.load(io.BytesIO(response.content))

# ---------- Load Model + Preprocessing ----------
@st.cache_resource
def load_model():
    model_url = "https://github.com/Anchita15/house_price_prediction_ML_pp/releases/download/v1.0/model.pkl"
    preprocess_url = "https://github.com/Anchita15/house_price_prediction_ML_pp/releases/download/v1.0/preprocess.pkl"
    model = download_from_github_release(model_url)
    preprocess = download_from_github_release(preprocess_url)
    return preprocess, model

preprocess, model = load_model()

# ---------- UI Layout ----------
st.markdown("<h1 style='text-align: center;'>🏠 House Price Prediction App</h1>", unsafe_allow_html=True)
st.write("Enter the details below to predict the house price:")

# Input fields
area = st.number_input("Area (sq ft)", min_value=500, max_value=10000, value=1000, step=100)
bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5])
bathrooms = st.selectbox("Bathrooms", [1, 2, 3])
stories = st.selectbox("Stories", [1, 2, 3])
mainroad = st.selectbox("Main Road", ["Yes", "No"])
guestroom = st.selectbox("Guest Room", ["Yes", "No"])
basement = st.selectbox("Basement", ["Yes", "No"])
hotwaterheating = st.selectbox("Hot Water Heating", ["Yes", "No"])
airconditioning = st.selectbox("Air Conditioning", ["Yes", "No"])
parking = st.selectbox("Parking Spots", [0, 1, 2, 3])
furnishingstatus = st.selectbox("Furnishing Status", ["Furnished", "Semi-Furnished", "Unfurnished"])

# Predict button
if st.button("📊 Predict Price"):
    input_data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'stories': [stories],
        'mainroad': [mainroad],
        'guestroom': [guestroom],
        'basement': [basement],
        'hotwaterheating': [hotwaterheating],
        'airconditioning': [airconditioning],
        'parking': [parking],
        'furnishingstatus': [furnishingstatus]
    })

    processed_input = preprocess.transform(input_data)
    prediction = model.predict(processed_input)[0]
    st.success(f"💰 Estimated House Price: ₹ {int(prediction):,}")
