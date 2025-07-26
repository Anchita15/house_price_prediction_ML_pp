# 🏡 House Price Prediction (California Housing Dataset)

This project builds a machine learning model to predict median house prices in California using census data. It includes complete data analysis, preprocessing, feature engineering, and model training using Linear Regression — all done in Google Colab.

---

## 📌 Problem Statement

Using the 1990 California census data, the goal is to predict the **median house value** for neighborhoods ("block groups") using features like population, median income, total rooms, and more.

---

## 📁 Dataset

- 📊 Source: [California Housing Dataset (Hands-On ML)](https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv)
- 🏘 20,640 entries
- 10 attributes (9 numerical, 1 categorical)

---

## 🧠 Key Features Used

- `median_income`
- `housing_median_age`
- `total_rooms`
- `population`
- `ocean_proximity` (categorical)

---

## 🧪 Data Preprocessing

- Handled missing values using **SimpleImputer**
- Created new features:
  - `rooms_per_household`
  - `bedrooms_per_room`
  - `population_per_household`
- Converted categorical column using **OneHotEncoder**
- Applied **StandardScaler** for normalization
- Used **Stratified Sampling** based on income categories to ensure fair train-test split
- Built preprocessing pipeline with **Pipeline** and **ColumnTransformer**

---

## 📊 EDA & Visualization

- Histograms for feature distributions
- Scatter plot of house prices by location (longitude vs latitude)
- Correlation matrix to find key predictors
- Used `scatter_matrix()` to visualize relationships between top features

---

## 🧮 Model Training

- Model: **LinearRegression** from Scikit-Learn
- Trained on processed data (`housing_prepared`)
- Evaluated predictions on a test batch

### 🧾 Sample Predictions:
The model was tested on a few instances from the training set. Below are the predicted vs. actual median house values:

| Row | Predicted ($) | Actual ($) |
|-----|---------------|------------|
| 1   | 85,658         | ~66,900     |
| 2   | 305,493        | ~303,900    |
| 3   | 152,056        | ~103,500    |
| 4   | 186,096        | ~146,600    |
| 5   | 244,551        | ~192,400    |

> 💡 These predictions demonstrate the model’s ability to generalize on real data fairly well, with Linear Regression capturing overall trends.


---

## 💡 Learnings

- How to build a full ML pipeline from scratch
- Importance of feature engineering
- Why stratified sampling improves model fairness
- How to analyze correlation and geospatial data

---

## 🔧 Tools & Libraries

- **Python** (Colab)
- **Pandas**, **NumPy**, **Matplotlib**
- **Scikit-Learn**: `Pipeline`, `ColumnTransformer`, `LinearRegression`, `StandardScaler`, etc.

---

## 📎 Project Highlights

✅ End-to-end pipeline built with reusable code  
✅ Real-world data cleaning and visualization  
✅ Clear model explainability using correlation and plots

---

> ⭐ Star this repo if you found it helpful, or feel free to fork and enhance with advanced models like Random Forest or XGBoost!
