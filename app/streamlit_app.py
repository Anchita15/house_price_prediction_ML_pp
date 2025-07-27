import streamlit as st
import joblib
import numpy as np

# App title
st.title("🏠 California House Price Predictor")

# User inputs (raw)
income = st.slider("Median Income", 0.0, 15.0, 3.0)
rooms = st.number_input("Total Rooms", value=2000)
bedrooms = st.number_input("Total Bedrooms", value=400)
population = st.number_input("Population", value=1000)
households = st.number_input("Households", value=500)
ocean = st.selectbox("Ocean Proximity", ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"])

#  Combine into one input sample (keep order same as `num_features + cat_features`)
features = np.array([[income, rooms, bedrooms, population, households, ocean]])

# Load the pipeline (which includes preprocessing + model)
model = joblib.load("model/pipeline.pkl")

# Predict
prediction = model.predict(features)

# Show result
st.write("### 💸 Predicted House Price: $", int(prediction[0]))
