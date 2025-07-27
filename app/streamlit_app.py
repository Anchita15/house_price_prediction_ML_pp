import streamlit as st
import joblib
import numpy as np

st.title("🏠 California House Price Predictor")

# User input
income = st.slider("Median Income", 0.0, 15.0, 3.0)
rooms = st.number_input("Total Rooms", value=2000)
bedrooms = st.number_input("Total Bedrooms", value=400)
population = st.number_input("Population", value=1000)
households = st.number_input("Households", value=500)

# Keep same order as training
features = np.array([[income, rooms, bedrooms, population, households]])

# Load model (relative path)
model = joblib.load("model/model.pkl")

# Predict
prediction = model.predict(features)

# Show result
st.write("💰 Predicted House Price: $", int(prediction[0]))
