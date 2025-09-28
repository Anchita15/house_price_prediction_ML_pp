import streamlit as st
import pandas as pd
import pickle

# Load model and preprocessing pipeline
@st.cache_data
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("preprocess.pkl", "rb") as f:
        preprocess = pickle.load(f)
    return preprocess, model

preprocess, model = load_model()

st.set_page_config(
    page_title="🏠 House Price Predictor",
    layout="wide",
    page_icon="🏡",
)

# 💫 Title Area
st.markdown(
    "<h1 style='text-align: center; color: #00f5d4;'>🏠 California House Price Predictor</h1>"
    "<p style='text-align: center; color: gray;'>Enter home features to get an instant price prediction</p>",
    unsafe_allow_html=True,
)

st.markdown("---")

# 📊 Sidebar Inputs
st.sidebar.title("✨ Enter House Details")

median_income = st.sidebar.slider("💰 Median Income (in $10,000s)", 0.0, 15.0, 5.0)
housing_median_age = st.sidebar.slider("🏗️ Median House Age", 1, 52, 20)
total_rooms = st.sidebar.number_input("🚪 Total Rooms", min_value=1, value=2000)
population = st.sidebar.number_input("👥 Population", min_value=1, value=3000)
ocean_proximity = st.sidebar.selectbox(
    "🌊 Ocean Proximity", ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
)

# 👇 Input DataFrame
input_data = pd.DataFrame([{
    "median_income": median_income,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "population": population,
    "ocean_proximity": ocean_proximity
}])

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🧾 Model Input Preview")
    st.dataframe(input_data, use_container_width=True)

with col2:
    st.markdown("### 🧠 Prediction Result")
    if st.button("💡 Predict House Price", use_container_width=True):
        X_proc = preprocess.transform(input_data)
        prediction = model.predict(X_proc)[0]
        st.success(f"🏡 **Estimated Price:** ${prediction:,.0f}")
