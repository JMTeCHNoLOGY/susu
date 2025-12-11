"""
Basic usage example of the Susu solar power prediction system with XGBoost.
"""

import sys
import os

# Add parent directory to path to import susu
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from susu import SolarPowerPredictor
import pandas as pd


def main():
    """Demonstrate basic usage of the solar power prediction system."""
    
    print("=" * 60)
    print("Solar Power Generation Prediction using XGBoost")
    print("=" * 60)
    
    # Create predictor instance
    predictor = SolarPowerPredictor()
    
    # Train the model with sample data
    print("\n1. Training the XGBoost model with sample data...")
    metrics = predictor.train(
        use_sample_data=True,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )
    
    # Display feature importance
    print("\n2. Feature Importance:")
    importance_df = predictor.get_feature_importance()
    print(importance_df.to_string(index=False))
    
    # Make sample predictions
    print("\n3. Making predictions on sample data...")
    sample_data = pd.DataFrame({
        'irradiance': [800, 500, 200],
        'temperature': [25, 30, 20],
        'humidity': [60, 70, 50],
        'wind_speed': [5, 8, 3],
        'hour': [12, 14, 16],
        'season': [1, 1, 2]
    })
    
    print("\nInput data:")
    print(sample_data)
    
    predictions = predictor.predict(sample_data)
    
    print("\nPredicted power output (kW):")
    for i, pred in enumerate(predictions):
        print(f"  Sample {i+1}: {pred:.2f} kW")
    
    # Optional: Save the model
    # print("\n4. Saving the model...")
    # predictor.save('solar_power_model.json')
    # print("Model saved successfully!")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
