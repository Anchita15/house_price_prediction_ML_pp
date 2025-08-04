import streamlit as st
import pandas as pd
import joblib
import warnings
import os

# Simple page configuration
st.title("🏠 House Price Predictor")

# Function to safely load model
def load_model_safely():
    """Load model with error handling"""
    try:
        if not os.path.exists("model/pipeline.pkl"):
            return None, "Model file not found at 'model/pipeline.pkl'"
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load("model/pipeline.pkl")
        return model, "Model loaded successfully!"
    
    except AttributeError as e:
        error_msg = f"Model compatibility issue: {str(e)}"
        return None, error_msg
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        return None, error_msg

# Load model
model, load_message = load_model_safely()

# Show loading status
if model is not None:
    st.success("✅ " + load_message)
else:
    st.error("❌ " + load_message)
    if "compatibility" in load_message:
        st.info("💡 This usually happens when the model was trained with a different sklearn version. Please retrain your model.")

# Create input form
st.subheader("Enter House Details")

# Input fields
longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-122.0, step=0.01)
latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=37.0, step=0.01)
rooms = st.number_input("Total Rooms", min_value=1, value=6)
bedrooms = st.number_input("Total Bedrooms", min_value=1, value=3)
population = st.number_input("Population", min_value=1, value=3000)
households = st.number_input("Number of Households", min_value=1, value=1000)
income = st.number_input("Median Income (10k USD)", min_value=0.0, value=5.0, step=0.01)
ocean = st.selectbox("Ocean Proximity", ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])

# Prediction button and logic
if st.button("🔍 Predict House Price"):
    if model is None:
        st.error("❌ Cannot make prediction: Model not loaded")
    else:
        try:
            # Calculate derived features
            rooms_per_household = rooms / households if households > 0 else 0
            bedrooms_per_room = bedrooms / rooms if rooms > 0 else 0
            population_per_household = population / households if households > 0 else 0
            
            # Create input dataframe
            data = pd.DataFrame({
                "longitude": [longitude],
                "latitude": [latitude],
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
            
            # Make prediction
            prediction = model.predict(data)
            predicted_price = int(prediction[0])
            
            # Display result
            st.success(f"💰 Predicted House Price: ${predicted_price:,}")
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")

# Simple troubleshooting info
st.markdown("---")
with st.expander("🛠️ Troubleshooting"):
    st.write("**If you see 'Model compatibility issue':**")
    st.write("1. The model was trained with a different sklearn version")
    st.write("2. Solution: Retrain your model with the current sklearn version")
    st.write("3. Or match the sklearn version used during training")
    
    st.code("""
# Quick fix - retrain your model:
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
# ... your training code ...
joblib.dump(pipeline, "model/pipeline.pkl")
    """)

st.markdown("*California Housing Price Predictor*")
