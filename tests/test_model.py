"""
Tests for the XGBoost model module.
"""

import unittest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from susu.model import SolarPowerModel
from susu.data_loader import DataLoader
import numpy as np


class TestSolarPowerModel(unittest.TestCase):
    """Test cases for SolarPowerModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = SolarPowerModel(n_estimators=10, random_state=42)
        
        # Generate sample data
        data_loader = DataLoader()
        data = data_loader.load_sample_data(n_samples=100)
        self.X_train, self.X_test, self.y_train, self.y_test = \
            data_loader.prepare_data(data, test_size=0.2)
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.model.model)
        self.assertIsNone(self.model.feature_importance)
    
    def test_train(self):
        """Test model training."""
        metrics = self.model.train(
            self.X_train, self.y_train,
            self.X_test, self.y_test,
            verbose=False
        )
        
        # Check that metrics are returned
        self.assertIn('train_rmse', metrics)
        self.assertIn('train_mae', metrics)
        self.assertIn('train_r2', metrics)
        self.assertIn('val_rmse', metrics)
        self.assertIn('val_mae', metrics)
        self.assertIn('val_r2', metrics)
        
        # Check that feature importance is set
        self.assertIsNotNone(self.model.feature_importance)
    
    def test_predict(self):
        """Test model prediction."""
        # Train the model first
        self.model.train(self.X_train, self.y_train, verbose=False)
        
        # Make predictions
        predictions = self.model.predict(self.X_test)
        
        # Check predictions shape
        self.assertEqual(predictions.shape[0], self.X_test.shape[0])
        
        # Check that predictions are non-negative
        self.assertTrue((predictions >= 0).all())
    
    def test_evaluate(self):
        """Test model evaluation."""
        # Train the model
        self.model.train(self.X_train, self.y_train, verbose=False)
        
        # Evaluate
        metrics = self.model.evaluate(self.X_test, self.y_test)
        
        # Check that metrics are returned
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)
        
        # Check that metrics are reasonable
        self.assertGreater(metrics['rmse'], 0)
        self.assertGreater(metrics['mae'], 0)
    
    def test_feature_importance(self):
        """Test feature importance retrieval."""
        # Train the model
        self.model.train(self.X_train, self.y_train, verbose=False)
        
        # Get feature importance
        importance = self.model.get_feature_importance()
        
        # Check that importance is returned
        self.assertIsNotNone(importance)
        self.assertEqual(len(importance), self.X_train.shape[1])


if __name__ == '__main__':
    unittest.main()
