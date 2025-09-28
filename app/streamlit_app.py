import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import io

# ---------- Load Model + Preprocessing ----------
@st.cache_resource
def load_model():
    def download_and_load(url):
        response = requests.get(url)
        response.raise_for_status()
        return pickle.load(io.BytesIO(response.content))

    model_url = "https://github.com/Anchita15/house_price_prediction_ML_pp/releases/download/v1.0/model.pkl"
    preprocess_url = "https://github.com/Anchita15/house_price_prediction_ML_pp/releases/download/v1.0/preprocess.pkl"

    model = download_and_load(model_url)
    preprocess = download_and_load(preprocess_url)
    return preprocess, model

preprocess, model = load_model()

# ---------- UI Layout ----------
st.set_page_config(page_title="🏠 House Price Predictor", layout="centered")
st.title("🏠 House Price Prediction App")
st.markdown("Enter the details below to predict the house price:")

# ---------- User Input ----------
area = st.number_input("Area (sq ft)", min_value=100, max_value=10000, value=1000)
bedrooms = st.selectbox("Bedrooms", options=[1, 2, 3, 4, 5])
bathrooms = st.selectbox("Bathrooms", options=[1, 2, 3, 4])
stories = st.selectbox("Stories", options=[1, 2, 3, 4])
mainroad = st.selectbox("Main Road", options=["Yes", "No"])
guestroom = st.selectbox("Guest Room", options=["Yes", "No"])
basement = st.selectbox("Basement", options=["Yes", "No"])
hotwaterheating = st.selectbox("Hot Water Heating", options=["Yes", "No"])
airconditioning = st.selectbox("Air Conditioning", options=["Yes", "No"])
parking = st.selectbox("Parking Spaces", options=[0, 1, 2, 3])
prefarea = st.selectbox("Preferred Area", options=["Yes", "No"])
furnishingstatus = st.selectbox("Furnishing Status", options=["Furnished", "Semi-Furnished", "Unfurnished"])

# ---------- DataFrame Creation ----------
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
    'prefarea': [prefarea],
    'furnishingstatus': [furnishingstatus]
})

# ---------- Prediction ----------
if st.button("Predict Price"):
    try:
        processed_input = preprocess.transform(input_data)
        prediction = model.predict(processed_input)[0]
        st.success(f"💰 Estimated House Price: ₹ {int(prediction):,}")
    except Exception as e:
        st.error("Something went wrong while predicting. Please check the inputs or try again.")
        st.exception(e)
