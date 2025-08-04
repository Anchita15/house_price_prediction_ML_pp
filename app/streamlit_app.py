"""
Quick script to download California housing data and train a compatible model
This will fix your Streamlit app immediately!
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import os

def download_and_prepare_data():
    """Download California housing dataset and prepare it"""
    print("📊 Downloading California housing dataset...")
    
    # Fetch the dataset
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    
    # The original dataset doesn't have ocean_proximity, so we'll create it
    # based on longitude/latitude (simplified approximation)
    def assign_ocean_proximity(row):
        long, lat = row['MedInc'], row['Latitude']  # Using available features
        if row['Longitude'] > -121.5:
            return 'INLAND'
        elif row['Latitude'] > 37.5:
            return 'NEAR BAY'
        elif row['Longitude'] < -123:
            return '<1H OCEAN'
        else:
            return 'NEAR OCEAN'
    
    df['ocean_proximity'] = df.apply(assign_ocean_proximity, axis=1)
    
    # Rename columns to match your app's expectations
    df = df.rename(columns={
        'MedInc': 'median_income',
        'HouseAge': 'housing_median_age',
        'AveRooms': 'total_rooms',
        'AveBedrms': 'total_bedrooms', 
        'Population': 'population',
        'AveOccup': 'households',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'MedHouseVal': 'median_house_value'
    })
    
    # Convert averages back to totals (approximate)
    df['total_rooms'] = (df['total_rooms'] * df['households']).round().astype(int)
    df['total_bedrooms'] = (df['total_bedrooms'] * df['households']).round().astype(int)
    
    # Add derived features
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']
    
    # Handle any infinite or NaN values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median())
    
    print(f"✅ Dataset prepared: {len(df)} samples")
    return df

def create_and_train_model(df):
    """Create and train the model pipeline"""
    print("🔧 Creating model pipeline...")
    
    # Define features (matching your Streamlit app exactly)
    feature_columns = [
        'longitude', 'latitude', 'total_rooms', 'total_bedrooms',
        'population', 'households', 'median_income', 'ocean_proximity',
        'rooms_per_household', 'bedrooms_per_room', 'population_per_household'
    ]
    
    X = df[feature_columns]
    y = df['median_house_value']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define feature types
    numerical_features = [
        'longitude', 'latitude', 'total_rooms', 'total_bedrooms',
        'population', 'households', 'median_income',
        'rooms_per_household', 'bedrooms_per_room', 'population_per_household'
    ]
    categorical_features = ['ocean_proximity']
    
    # Create preprocessor (avoiding the problematic 'passthrough')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'  # This avoids the _RemainderColsList issue
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
    
    print("🚀 Training model...")
    pipeline.fit(X_train, y_train)
    
    # Quick evaluation
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    print(f"📈 Model Performance:")
    print(f"   Training R²: {train_score:.3f}")
    print(f"   Test R²: {test_score:.3f}")
    
    return pipeline, X_test

def save_model(pipeline, X_test):
    """Save the trained model"""
    print("💾 Saving model...")
    
    # Create model directory
    os.makedirs('model', exist_ok=True)
    
    # Save the model
    joblib.dump(pipeline, 'model/pipeline.pkl')
    
    # Test the saved model
    print("🧪 Testing saved model...")
    loaded_model = joblib.load('model/pipeline.pkl')
    
    # Make a test prediction
    sample = X_test.iloc[0:1]
    prediction = loaded_model.predict(sample)
    
    print(f"✅ Test prediction: ${prediction[0]:,.0f}")
    print("🎉 Model saved successfully!")
    
    return True

def main():
    """Main function to fix your Streamlit app"""
    print("🏠 QUICK FIX: California House Price Model")
    print("=" * 50)
    
    try:
        # Download and prepare data
        df = download_and_prepare_data()
        
        # Train model
        pipeline, X_test = create_and_train_model(df)
        
        # Save model
        save_model(pipeline, X_test)
        
        print("\n🎊 SUCCESS! Your Streamlit app should now work!")
        print("🔄 Refresh your Streamlit app to see the changes.")
        print("\n💡 What happened:")
        print("   ✅ Downloaded California housing dataset")
        print("   ✅ Created features matching your app")
        print("   ✅ Trained model with current sklearn version")
        print("   ✅ Saved compatible model to model/pipeline.pkl")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check the error and try again.")

if __name__ == "__main__":
    main()
