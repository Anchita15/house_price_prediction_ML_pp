import pandas as pd
import joblib
import streamlit as st

# Collect user input
rooms = st.number_input("Total Rooms", min_value=1)
bedrooms = st.number_input("Total Bedrooms", min_value=1)
population = st.number_input("Population", min_value=1)
households = st.number_input("Number of Households", min_value=1)
income = st.number_input("Median Income (10k USD)", min_value=0.0, step=0.01)
ocean = st.selectbox("Ocean Proximity", ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])

if st.button("🔍 Predict House Price"):
    # Add derived features
    rooms_per_household = rooms / households
    bedrooms_per_room = bedrooms / rooms
    population_per_household = population / households

    # Create full dataframe
    data = pd.DataFrame({
        "total_rooms": [rooms],
        "total_bedrooms": [bedrooms],
        "population": [population],
        "households": [households],
        "median_income": [income],
        "ocean_proximity": [ocean],
        "rooms_per_household": [rooms_per_household],
        "bedrooms_per_room": [bedrooms_per_room],
        "population_per_household": [population_per_household]
    })

    # Load model and predict
    model = joblib.load("model/pipeline.pkl")
    prediction = model.predict(data)
    st.success(f"💰 Predicted House Price: ${int(prediction[0])}")
