import streamlit as st
import joblib
import numpy as np

st.title("🏠 California House Price Predictor")

# 📥 User Inputs
income = st.slider("Median Income", 0.0, 15.0, 3.0)
rooms = st.number_input("Total Rooms", value=2000)
bedrooms = st.number_input("Total Bedrooms", value=400)
population = st.number_input("Population", value=1000)
households = st.number_input("Households", value=500)
ocean_proximity = st.selectbox("Ocean Proximity", ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"])

# 🧠 Model Prediction
features = np.array([[income, rooms, bedrooms, population, households, ocean_proximity]])

model = joblib.load("model/pipeline.pkl")  # ✅ updated to pipeline
prediction = model.predict(features)

st.write("💰 Predicted House Price: $", int(prediction[0]))
