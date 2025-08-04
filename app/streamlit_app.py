import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Title
st.title("California Housing Prices Explorer")

# Load data
house_price_url = 'https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv'
prices = pd.read_csv(house_price_url)

# Show raw data
st.subheader("Raw Data Preview")
st.write(prices.head())

# Dataset info
st.subheader("Dataset Info")
buffer = []
prices.info(buf := buffer)
st.text('\n'.join(buf))

# Value counts for categorical column
st.subheader("Ocean Proximity Distribution")
st.bar_chart(prices['ocean_proximity'].value_counts())

# Histogram of all numerical features
st.subheader("Histograms of Numeric Columns")
fig, ax = plt.subplots(figsize=(10, 8))
prices.hist(bins=50, figsize=(10, 8), ax=ax)
st.pyplot(fig)

# Train-test split
st.subheader("Train-Test Split")
train_set, test_set = train_test_split(prices, test_size=0.2, random_state=42)
st.success(f"Train set size: {len(train_set)}\nTest set size: {len(test_set)}")
