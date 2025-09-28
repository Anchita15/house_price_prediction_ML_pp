import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import requests

# ---------- Helper: Download large file from Google Drive ----------
def download_from_drive(file_id):
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)

    def get_confirm_token(resp):
        for key, value in resp.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    token = get_confirm_token(response)
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    file_data = io.BytesIO()
    for chunk in response.iter_content(32768):
        if chunk:
            file_data.write(chunk)
    file_data.seek(0)

    return pickle.load(file_data)

# ---------- Load Model + Preprocessing ----------
@st.cache_resource
def load_model():
    model = download_from_drive("14hE4hgNECHC1GL6DWzaCf3YUyyLqLBiN")
    preprocess = download_from_drive("16OtGYOdr3NH8Ni1WSK9GPWe423G_HhaM")
    return preprocess, model

preprocess, model = load_model()

# ---------- UI Layout ----------
st.set_page_config(page_title="🏠 House Price Predictor", layout="centered")
st.title("🏠 House Price Prediction App")
st.caption("Built with ML, Streamlit & Google Drive by [@Anchita15](https://github.com/Anchita15)")

st.markdown("---")

# ---------- Input Fields ----------
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq ft)", min_value=200, max_value=10000, value=1200)
    bedrooms = st.slider("Bedrooms", 1, 5, 3)
    bathrooms = st.slider("Bathrooms", 1, 4, 2)

with col2:
    stories = st.selectbox("Number of Stories", [1, 2, 3, 4], index=1)
    parking = st.slider("Parking Spaces", 0, 3, 1)
    mainroad = st.selectbox("On Main Road?", ["yes", "no"])
    guestroom = st.selectbox("Guest Room?", ["yes", "no"])

# ---------- Prediction ----------
if st.button("💰 Predict Price"):
    input_data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'stories': [stories],
        'parking': [parking],
        'mainroad': [mainroad],
        'guestroom': [guestroom]
    })

    X_transformed = preprocess.transform(input_data)
    prediction = model.predict(X_transformed)[0]

    st.success(f"🏷️ Estimated Price: ₹ {int(prediction):,}")

# ---------- Footer ----------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; font-size: 14px;'>"
    "🚀 Made with ❤️ using Streamlit & XGBoost"
    "</div>",
    unsafe_allow_html=True
)
