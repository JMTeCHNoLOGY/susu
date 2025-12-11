"""
Tests for the predictor module.
"""

import unittest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from susu.predictor import SolarPowerPredictor
import pandas as pd
import numpy as np


class TestSolarPowerPredictor(unittest.TestCase):
    """Test cases for SolarPowerPredictor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.predictor = SolarPowerPredictor()
    
    def test_initialization(self):
        """Test predictor initialization."""
        self.assertIsNotNone(self.predictor.data_loader)
        self.assertIsNone(self.predictor.model)
        self.assertFalse(self.predictor.is_trained)
    
    def test_create_model(self):
        """Test model creation."""
        self.predictor.create_model(n_estimators=10)
        self.assertIsNotNone(self.predictor.model)
    
    def test_train_with_sample_data(self):
        """Test training with sample data."""
        metrics = self.predictor.train(
            use_sample_data=True,
            n_estimators=10,
            verbose=False
        )
        
        # Check that training completed
        self.assertTrue(self.predictor.is_trained)
        
        # Check that metrics are returned
        self.assertIn('train_rmse', metrics)
        self.assertIn('val_rmse', metrics)
    
    def test_predict(self):
        """Test prediction."""
        # Train the model
        self.predictor.train(use_sample_data=True, n_estimators=10, verbose=False)
        
        # Create sample input
        sample_data = pd.DataFrame({
            'irradiance': [800, 500],
            'temperature': [25, 30],
            'humidity': [60, 70],
            'wind_speed': [5, 8],
            'hour': [12, 14],
            'season': [1, 1]
        })
        
        # Make predictions
        predictions = self.predictor.predict(sample_data)
        
        # Check predictions
        self.assertEqual(len(predictions), 2)
        self.assertTrue((predictions >= 0).all())
    
    def test_get_feature_importance(self):
        """Test feature importance retrieval."""
        # Train the model
        self.predictor.train(use_sample_data=True, n_estimators=10, verbose=False)
        
        # Get feature importance
        importance_df = self.predictor.get_feature_importance()
        
        # Check that importance is returned as DataFrame
        self.assertIsInstance(importance_df, pd.DataFrame)
        self.assertIn('feature', importance_df.columns)
        self.assertIn('importance', importance_df.columns)
    
    def test_predict_before_training_raises_error(self):
        """Test that predicting before training raises an error."""
        sample_data = pd.DataFrame({
            'irradiance': [800],
            'temperature': [25],
            'humidity': [60],
            'wind_speed': [5],
            'hour': [12],
            'season': [1]
        })
        
        with self.assertRaises(RuntimeError):
            self.predictor.predict(sample_data)


if __name__ == '__main__':
    unittest.main()
