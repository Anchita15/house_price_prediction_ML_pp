import streamlit as st
import joblib
import numpy as np
import sklearn
import sklearn.compose._column_transformer as ct

# 🩹 HOTFIX for loading pipeline.pkl created in Colab with sklearn 1.6.x
class _RemainderColsList(list):
    pass
ct._RemainderColsList = _RemainderColsList

st.title("🏠 California House Price Predictor")

income = st.slider("Median Income", 0.0, 15.0, 3.0)
rooms = st.number_input("Total Rooms", value=2000)
bedrooms = st.number_input("Total Bedrooms", value=400)
population = st.number_input("Population", value=1000)
households = st.number_input("Households", value=500)

features = np.array([[income, rooms, bedrooms, population, households]])

# 👇 Load the pipeline model AFTER hotfix
model = joblib.load("model/pipeline.pkl")
prediction = model.predict(features)

st.write("💰 Predicted House Price: $", int(prediction[0]))
