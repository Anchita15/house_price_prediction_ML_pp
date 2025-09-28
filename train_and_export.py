import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

# --- Load the data ---
df = pd.read_csv("prices.csv")

# --- Drop rows with missing target ---
df.dropna(subset=["median_house_value"], inplace=True)

# --- Define features and target ---
X = df[["housing_median_age", "total_rooms", "population", "median_income", "ocean_proximity"]]
y = df["median_house_value"]

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Preprocessing ---
numeric_features = ["housing_median_age", "total_rooms", "population", "median_income"]
categorical_features = ["ocean_proximity"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# --- Model pipeline ---
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# --- Fit the model ---
model_pipeline.fit(X_train, y_train)

# --- Save model and preprocessor ---
app_dir = "app"
os.makedirs(app_dir, exist_ok=True)

with open(f"{app_dir}/model.pkl", "wb") as f:
    pickle.dump(model_pipeline.named_steps["regressor"], f)

with open(f"{app_dir}/preprocess.pkl", "wb") as f:
    pickle.dump(model_pipeline.named_steps["preprocessor"], f)

print("✅ Saved model.pkl and preprocess.pkl to /app folder!")
