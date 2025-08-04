import streamlit as st
import pandas as pd
import joblib
import sklearn
import warnings
from pathlib import Path

# Configure the page
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 California House Price Predictor")
st.markdown("Predict house prices based on location and property features")

# Display sklearn version for debugging
with st.expander("🔧 System Info"):
    st.write(f"Scikit-learn version: {sklearn.__version__}")
    st.write(f"Pandas version: {pd.__version__}")

# Function to safely load model
@st.cache_resource
def load_model():
    """Load the trained model with error handling"""
    try:
        model_path = Path("model/pipeline.pkl")
        if not model_path.exists():
            st.error("❌ Model file not found at 'model/pipeline.pkl'")
            return None
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load(model_path)
        
        st.success("✅ Model loaded successfully!")
        return model
    
    except AttributeError as e:
        st.error("❌ Model compatibility issue detected!")
        st.error("This usually happens when the model was trained with a different sklearn version.")
        st.info("💡 **Solution**: Retrain your model with the current sklearn version or match the sklearn version used during training.")
        st.code(f"Error details: {str(e)}")
        return None
    
    except Exception as e:
        st.error(f"❌ Unexpected error loading model: {str(e)}")
        return None

# Load the model
model = load_model()

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Location Features")
    longitude = st.number_input(
        "Longitude", 
        min_value=-180.0, 
        max_value=180.0, 
        value=-122.0,
        step=0.01,
        help="Geographic longitude coordinate"
    )
    latitude = st.number_input(
        "Latitude", 
        min_value=-90.0, 
        max_value=90.0, 
        value=37.0,
        step=0.01,
        help="Geographic latitude coordinate"
    )
    ocean = st.selectbox(
        "Ocean Proximity", 
        ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
        help="Distance to ocean"
    )

with col2:
    st.subheader("🏘️ Property Features")
    rooms = st.number_input(
        "Total Rooms", 
        min_value=1, 
        value=6,
        help="Total number of rooms in the house"
    )
    bedrooms = st.number_input(
        "Total Bedrooms", 
        min_value=1, 
        value=3,
        help="Total number of bedrooms"
    )
    population = st.number_input(
        "Population", 
        min_value=1, 
        value=3000,
        help="Population in the area"
    )
    households = st.number_input(
        "Number of Households", 
        min_value=1, 
        value=1000,
        help="Number of households in the area"
    )
    income = st.number_input(
        "Median Income (10k USD)", 
        min_value=0.0, 
        value=5.0,
        step=0.01,
        help="Median income in tens of thousands of dollars"
    )

# Validation
def validate_inputs():
    """Validate user inputs"""
    errors = []
    
    if bedrooms > rooms:
        errors.append("❌ Total bedrooms cannot exceed total rooms")
    
    if households > population:
        errors.append("❌ Number of households cannot exceed population")
    
    # Check for reasonable ratios
    if rooms / households > 50:
        errors.append("⚠️ Warning: Very high rooms per household ratio")
    
    if population / households > 10:
        errors.append("⚠️ Warning: Very high population per household ratio")
    
    return errors

# Prediction section
st.subheader("🔮 Make Prediction")

# Show validation errors
validation_errors = validate_inputs()
if validation_errors:
    for error in validation_errors:
        st.warning(error)

# Center the predict button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔍 Predict House Price", type="primary", use_container_width=True)

if predict_button:
    if model is None:
        st.error("❌ Cannot make prediction: Model not loaded")
        st.info("Please fix the model loading issue first.")
    
    elif validation_errors and any("❌" in error for error in validation_errors):
        st.error("❌ Please fix the input validation errors before predicting")
    
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
            with st.spinner("Making prediction..."):
                prediction = model.predict(data)
                predicted_price = int(prediction[0])
            
            # Display result
            st.success(f"💰 **Predicted House Price: ${predicted_price:,}**")
            
            # Show additional insights
            with st.expander("📊 Prediction Details"):
                st.write("**Input Features:**")
                feature_df = pd.DataFrame({
                    'Feature': [
                        'Longitude', 'Latitude', 'Total Rooms', 'Total Bedrooms',
                        'Population', 'Households', 'Median Income', 'Ocean Proximity',
                        'Rooms per Household', 'Bedrooms per Room', 'Population per Household'
                    ],
                    'Value': [
                        longitude, latitude, rooms, bedrooms, population, households, 
                        f"${income*10000:,}", ocean, f"{rooms_per_household:.2f}", 
                        f"{bedrooms_per_room:.2f}", f"{population_per_household:.2f}"
                    ]
                })
                st.dataframe(feature_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.info("This might be due to model compatibility issues or unexpected input format.")

# Add troubleshooting section
with st.expander("🛠️ Troubleshooting"):
    st.markdown("""
    **Common Issues:**
    
    1. **AttributeError: Can't get attribute '_RemainderColsList'**
       - This happens when the model was trained with a different sklearn version
       - Solution: Retrain your model with the current sklearn version
    
    2. **Model file not found**
       - Ensure 'model/pipeline.pkl' exists in your project directory
       - Check file permissions
    
    3. **Prediction errors**
       - Verify input data format matches training data
       - Check for missing or invalid values
    
    **Model Retraining Code:**
    ```python
    # Example code to retrain your model
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline
    
    # Recreate your pipeline and retrain
    # ... your training code here ...
    
    # Save with current sklearn version
    joblib.dump(pipeline, "model/pipeline.pkl")
    ```
    """)

# Footer
st.markdown("---")
st.markdown("*California Housing Price Predictor - Built with Streamlit*")
