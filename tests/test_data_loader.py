"""
Tests for the data loader module.
"""

import unittest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from susu.data_loader import DataLoader
import numpy as np
import pandas as pd


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_loader = DataLoader()
    
    def test_load_sample_data(self):
        """Test sample data generation."""
        data = self.data_loader.load_sample_data(n_samples=100)
        
        # Check that data is a DataFrame
        self.assertIsInstance(data, pd.DataFrame)
        
        # Check number of samples
        self.assertEqual(len(data), 100)
        
        # Check that required columns exist
        required_columns = ['irradiance', 'temperature', 'humidity', 
                          'wind_speed', 'hour', 'season', 'power_output']
        for col in required_columns:
            self.assertIn(col, data.columns)
        
        # Check that values are in reasonable ranges
        self.assertTrue((data['irradiance'] >= 0).all())
        self.assertTrue((data['irradiance'] <= 1000).all())
        self.assertTrue((data['temperature'] >= 15).all())
        self.assertTrue((data['temperature'] <= 45).all())
        self.assertTrue((data['power_output'] >= 0).all())
    
    def test_prepare_data(self):
        """Test data preparation and splitting."""
        data = self.data_loader.load_sample_data(n_samples=100)
        X_train, X_test, y_train, y_test = self.data_loader.prepare_data(
            data, test_size=0.2
        )
        
        # Check shapes
        self.assertEqual(X_train.shape[0], 80)
        self.assertEqual(X_test.shape[0], 20)
        self.assertEqual(y_train.shape[0], 80)
        self.assertEqual(y_test.shape[0], 20)
        
        # Check that features are scaled (mean close to 0, std close to 1)
        self.assertTrue(np.abs(X_train.mean()) < 1.0)
    
    def test_transform(self):
        """Test data transformation."""
        data = self.data_loader.load_sample_data(n_samples=100)
        X_train, X_test, _, _ = self.data_loader.prepare_data(data)
        
        # Create new data
        new_data = data.drop(columns=['power_output']).iloc[:5]
        transformed = self.data_loader.transform(new_data)
        
        # Check shape
        self.assertEqual(transformed.shape[0], 5)
        self.assertEqual(transformed.shape[1], X_train.shape[1])


if __name__ == '__main__':
    unittest.main()
