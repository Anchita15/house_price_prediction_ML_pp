import streamlit as st
import pandas as pd
import pickle
import requests
from io import BytesIO

# ---------------- Page Config ----------------
st.set_page_config(page_title="🏠 House Price Predictor", layout="centered")
st.markdown("<h1 style='text-align: center;'>🏠 House Price Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter the details below to predict the house price:</p>", unsafe_allow_html=True)

# ---------------- Load Model & Preprocessor ----------------
@st.cache_resource
def load_pickle_from_github(url):
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Failed to download file.")
        return None
    return pickle.load(BytesIO(response.content))

model_url = "https://github.com/Anchita15/house_price_prediction_ml_pp/releases/download/v1.0/model.pkl"
preprocess_url = "https://github.com/Anchita15/house_price_prediction_ml_pp/releases/download/v1.0/preprocess.pkl"

model = load_pickle_from_github(model_url)
preprocessor = load_pickle_from_github(preprocess_url)

# ---------------- Input Form ----------------
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        income = st.slider("💰 Median Income (in $10,000s)", 0.0, 15.0, 5.0)
        age = st.slider("🏡 Median House Age", 1, 50, 20)
        rooms = st.number_input("🛏 Total Rooms", min_value=1, max_value=10000, value=2000)

    with col2:
        population = st.number_input("👥 Population", min_value=1, max_value=50000, value=3000)
        proximity = st.selectbox("🌊 Ocean Proximity", ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])

    submitted = st.form_submit_button("🔍 Predict House Price")

# ---------------- Prediction ----------------
if submitted:
    try:
        input_df = pd.DataFrame([{
            'median_income': income,
            'housing_median_age': age,
            'total_rooms': rooms,
            'population': population,
            'ocean_proximity': proximity
        }])

        # Ensure correct column order
        expected_columns = ['median_income', 'housing_median_age', 'total_rooms', 'population', 'ocean_proximity']
        input_df = input_df[expected_columns]

        # Preprocess and predict
        X_processed = preprocessor.transform(input_df)
        prediction = model.predict(X_processed)[0]

        st.success(f"🏠 Estimated House Price: **${prediction:,.2f}**")

        st.subheader("🧾 Model Input Preview")
        st.dataframe(input_df)

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")