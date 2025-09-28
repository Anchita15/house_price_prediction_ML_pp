import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import io

st.set_page_config(
    page_title="House Price Prediction",
    layout="centered",
    initial_sidebar_state="auto"
)

# Apply a nice dark mode-friendly style
st.markdown("""
    <style>
        html, body, [class*="css"] {
            background-color: #0e1117;
            color: #fafafa;
        }
        .stTextInput>div>div>input, .stSelectbox>div>div>div>input {
            background-color: #262730;
            color: white;
        }
        .stButton>button {
            background-color: #1f77b4;
            color: white;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)


# ---------- Load Model from Google Drive ----------
@st.cache_data
def load_model():
    def download_from_drive(file_id: str):
        url = f"https://drive.google.com/uc?id={file_id}"
        resp = requests.get(url)
        resp.raise_for_status()
        return pickle.load(io.BytesIO(resp.content))

    model = download_from_drive("14hE4hgNECHC1GL6DWzaCf3YUyyLqLBiN")         # model.pkl
    preprocess = download_from_drive("16OtGYOdr3NH8Ni1WSK9GPWe423G_HhaM")    # preprocess.pkl
    return preprocess, model


preprocess, model = load_model()


# ---------- UI Layout ----------
st.title("🏡 House Price Prediction App")

st.markdown("Fill in the property details below to get an estimated price.")

# Example Inputs — You can modify these based on your dataset features
area = st.number_input("Area (sq ft)", min_value=500, max_value=10000, value=1500)
bedrooms = st.slider("Number of Bedrooms", 1, 10, 3)
bathrooms = st.slider("Number of Bathrooms", 1, 10, 2)
stories = st.selectbox("Number of Stories", [1, 2, 3, 4])
mainroad = st.selectbox("Is Main Road Facing?", ["Yes", "No"])
guestroom = st.selectbox("Guest Room Available?", ["Yes", "No"])
basement = st.selectbox("Has Basement?", ["Yes", "No"])
hotwaterheating = st.selectbox("Hot Water Heating?", ["Yes", "No"])
airconditioning = st.selectbox("Air Conditioning?", ["Yes", "No"])
parking = st.slider("Parking Spaces", 0, 5, 1)
prefarea = st.selectbox("Preferred Area?", ["Yes", "No"])
furnishing = st.selectbox("Furnishing Status", ["Furnished", "Semi-Furnished", "Unfurnished"])

if st.button("Predict Price 💰"):
    try:
        # Create DataFrame for preprocessing
        input_df = pd.DataFrame({
            "area": [area],
            "bedrooms": [bedrooms],
            "bathrooms": [bathrooms],
            "stories": [stories],
            "mainroad": [mainroad],
            "guestroom": [guestroom],
            "basement": [basement],
            "hotwaterheating": [hotwaterheating],
            "airconditioning": [airconditioning],
            "parking": [parking],
            "prefarea": [prefarea],
            "furnishingstatus": [furnishing]
        })

        transformed = preprocess.transform(input_df)
        prediction = model.predict(transformed)[0]

        st.success(f"🏷️ Estimated House Price: ₹ {np.round(prediction, 2):,.2f}")

    except Exception as e:
        st.error("⚠️ An error occurred while predicting. Please check input and try again.")
        st.exception(e)
