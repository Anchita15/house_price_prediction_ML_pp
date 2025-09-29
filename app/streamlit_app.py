import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import io

st.set_page_config(page_title="🏠 House Price Predictor", layout="centered")

# ---------- Load pickled files from GitHub Release ----------
@st.cache_resource
def load_model():
    def load_from_url(url):
        response = requests.get(url)
        response.raise_for_status()
        return pickle.load(io.BytesIO(response.content))

    model_url = "https://github.com/Anchita15/house_price_prediction_ML_pp/releases/download/v1.0/model.pkl"
    preprocess_url = "https://github.com/Anchita15/house_price_prediction_ML_pp/releases/download/v1.0/preprocess.pkl"
    
    model = load_from_url(model_url)
    preprocess = load_from_url(preprocess_url)
    
    return preprocess, model

# Load models
try:
    preprocess, model = load_model()
except Exception as e:
    st.error("❌ Failed to load model or preprocess file.")
    st.stop()

# ---------- UI ----------
st.markdown("<h1 style='text-align: center;'>🏠 House Price Prediction App</h1>", unsafe_allow_html=True)
st.markdown("Enter the details below to predict the house price:")

# Inputs
area = st.number_input("Area (sq ft)", min_value=500, max_value=10000, value=1000)
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

# Prediction
if st.button("📊 Predict Price"):
    try:
        input_df = pd.DataFrame({
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

        # Show preview
        st.markdown("### 🧾 Input Summary")
        st.dataframe(input_df)

        # Transform & predict
        transformed = preprocess.transform(input_df)
        price = model.predict(transformed)[0]
        st.success(f"💰 Estimated House Price: ₹ {int(price):,}")
    
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
