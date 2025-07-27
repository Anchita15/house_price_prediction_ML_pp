import streamlit as st
import joblib
import numpy as np

# App title
st.title("🏠 California House Price Predictor")

# 📥 User Inputs - must match training feature order:
income = st.slider("Median Income", 0.0, 15.0, 3.0)
rooms = st.number_input("Total Rooms", value=2000)
age = st.slider("Housing Median Age", 1, 100, 30)  # ✅ Added
households = st.number_input("Households", value=500)
ocean = st.selectbox("Ocean Proximity", ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"])

# ✅ Order = [median_income, total_rooms, housing_median_age, households, ocean_proximity]
features = np.array([[income, rooms, age, households, ocean]])

# 🔄 Load model (pipeline = preprocessing + model)
model = joblib.load("model/pipeline.pkl")

# 🎯 Predict
prediction = model.predict(features)

# 💸 Display
st.write("### 💸 Predicted House Price: $", int(prediction[0]))
