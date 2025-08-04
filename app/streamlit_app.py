"""
Script to retrain the house price prediction model with current sklearn version
This will fix the compatibility issue and allow your Streamlit app to work properly.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import os

def create_derived_features(df):
    """Add derived features to the dataset"""
    df = df.copy()
    
    # Avoid division by zero
    df['rooms_per_household'] = df['total_rooms'] / df['households'].replace(0, np.nan)
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms'].replace(0, np.nan)
    df['population_per_household'] = df['population'] / df['households'].replace(0, np.nan)
    
    # Fill any NaN values with median
    df['rooms_per_household'].fillna(df['rooms_per_household'].median(), inplace=True)
    df['bedrooms_per_room'].fillna(df['bedrooms_per_room'].median(), inplace=True)
    df['population_per_household'].fillna(df['population_per_household'].median(), inplace=True)
    
    return df

def load_and_prepare_data():
    """
    Load your training data here.
    Replace this section with your actual data loading code.
    """
    # Example - replace with your actual data loading
    try:
        # Try to load from common locations
        data_paths = [
            'data/housing.csv',
            'housing.csv',
            'california_housing.csv',
            'train.csv'
        ]
        
        df = None
        for path in data_paths:
            if os.path.exists(path):
                print(f"Loading data from {path}")
                df = pd.read_csv(path)
                break
        
        if df is None:
            print("No data file found. Please update the data loading section.")
            print("Expected columns: longitude, latitude, housing_median_age, total_rooms,")
            print("total_bedrooms, population, households, median_income, median_house_value, ocean_proximity")
            return None, None, None, None
        
        # Add derived features
        df = create_derived_features(df)
        
        # Define features and target
        feature_columns = [
            'longitude', 'latitude', 'total_rooms', 'total_bedrooms',
            'population', 'households', 'median_income', 'ocean_proximity',
            'rooms_per_household', 'bedrooms_per_room', 'population_per_household'
        ]
        
        # Check if target column exists (common names)
        target_columns = ['median_house_value', 'house_value', 'price', 'target']
        target_col = None
        for col in target_columns:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            print(f"Target column not found. Available columns: {list(df.columns)}")
            return None, None, None, None
        
        X = df[feature_columns]
        y = df[target_col]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Data loaded successfully!")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Features: {len(feature_columns)}")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

def create_pipeline():
    """Create the ML pipeline"""
    
    # Define feature types
    numerical_features = [
        'longitude', 'latitude', 'total_rooms', 'total_bedrooms',
        'population', 'households', 'median_income',
        'rooms_per_household', 'bedrooms_per_room', 'population_per_household'
    ]
    categorical_features = ['ocean_proximity']
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'  # This replaces the problematic 'passthrough'
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    return pipeline

def train_and_save_model():
    """Main function to train and save the model"""
    
    print("🏠 California House Price Prediction - Model Training")
    print("=" * 55)
    
    # Load data
    print("📊 Loading data...")
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    if X_train is None:
        print("❌ Failed to load data. Please check your data file.")
        print("\n💡 To use this script:")
        print("1. Place your housing dataset in one of these locations:")
        print("   - data/housing.csv")
        print("   - housing.csv")
        print("   - california_housing.csv")
        print("2. Ensure it has the required columns")
        print("3. Run this script again")
        return
    
    # Create pipeline
    print("🔧 Creating ML pipeline...")
    pipeline = create_pipeline()
    
    # Train model
    print("🚀 Training model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate model
    print("📈 Evaluating model...")
    train_predictions = pipeline.predict(X_train)
    test_predictions = pipeline.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, train_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)
    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    
    print(f"\n📊 Model Performance:")
    print(f"   Training MAE: ${train_mae:,.0f}")
    print(f"   Test MAE: ${test_mae:,.0f}")
    print(f"   Training R²: {train_r2:.3f}")
    print(f"   Test R²: {test_r2:.3f}")
    
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Save model
    print("💾 Saving model...")
    joblib.dump(pipeline, 'model/pipeline.pkl')
    
    print("✅ Model training completed successfully!")
    print("🎉 Your Streamlit app should now work properly!")
    
    # Test the saved model
    print("\n🧪 Testing saved model...")
    try:
        loaded_model = joblib.load('model/pipeline.pkl')
        test_prediction = loaded_model.predict(X_test.iloc[:1])
        print(f"   Test prediction: ${test_prediction[0]:,.0f}")
        print("✅ Model loads and predicts successfully!")
    except Exception as e:
        print(f"❌ Error testing saved model: {e}")

if __name__ == "__main__":
    train_and_save_model()
