import streamlit as st

st.title("🏠 House Price Predictor - Test")
st.write("If you can see this, Streamlit is working!")

# Test the imports
try:
    import pandas as pd
    st.success("✅ Pandas imported successfully")
except Exception as e:
    st.error(f"❌ Pandas import failed: {e}")

try:
    import joblib
    st.success("✅ Joblib imported successfully")
except Exception as e:
    st.error(f"❌ Joblib import failed: {e}")

# Simple input test
test_input = st.number_input("Test Input", value=1.0)
if st.button("Test Button"):
    st.write(f"Button clicked! Input value: {test_input}")
