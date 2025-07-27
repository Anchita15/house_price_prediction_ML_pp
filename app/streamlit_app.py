import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="🏠 House Price Predictor", layout="centered")
st.title("🏡 California House Price Prediction")
st.markdown("Enter the details below to predict the **median house price**.")

# 🌟 Input fields
rooms = st.number_input("Total Rooms", min_value=1)
bedrooms = st.number_input("Total Bedrooms", min_value=1)
population = st.number_input("Population", min_value=1)
households = st.number_input("Number of Households", min_value=1)
income = st.number_input("Median Income (10k USD)", min_value=0.0, step=0.1)
ocean = st.selectbox("Ocean Proximity", 
                     ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])

# Predict button
if st.button("🔍 Predict House Price"):

    try:
        # 🔧 Prepare DataFrame with original + engineered features
        data = pd.DataFrame({
            "total_rooms": [rooms],
            "total_bedrooms": [bedrooms],
            "population": [population],
            "households": [households],
            "median_income": [income],
            "ocean_proximity": [ocean]
        })

        # ⚙️ Add engineered features
        data["rooms_per_household"] = data["total_rooms"] / data["households"]
        data["bedrooms_per_room"] = data["total_bedrooms"] / data["total_rooms"]
        data["population_per_household"] = data["population"] / data["households"]

        # 🔍 Load trained pipeline and make prediction
        model = joblib.load("model/pipeline.pkl")
        prediction = model.predict(data)

        st.success(f"💰 Predicted House Price: ${int(prediction[0])}")
    
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
