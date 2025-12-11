"""
Data loading and preprocessing module for solar power generation data.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataLoader:
    """
    Handles loading and preprocessing of solar power generation data.
    """
    
    def __init__(self):
        """Initialize the DataLoader."""
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def load_sample_data(self, n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
        """
        Generate sample solar power generation data for demonstration.
        
        Args:
            n_samples: Number of samples to generate
            random_state: Random seed for reproducibility
            
        Returns:
            DataFrame with solar power generation features and target
        """
        np.random.seed(random_state)
        
        # Generate synthetic features
        # Solar irradiance (W/m²)
        irradiance = np.random.uniform(0, 1000, n_samples)
        
        # Temperature (°C)
        temperature = np.random.uniform(15, 45, n_samples)
        
        # Humidity (%)
        humidity = np.random.uniform(20, 90, n_samples)
        
        # Wind speed (m/s)
        wind_speed = np.random.uniform(0, 15, n_samples)
        
        # Time of day (0-23 hours)
        hour = np.random.randint(0, 24, n_samples)
        
        # Season (0-3: spring, summer, fall, winter)
        season = np.random.randint(0, 4, n_samples)
        
        # Generate target variable (power output in kW)
        # Power is highly correlated with irradiance, with some temperature effect
        power_output = (
            irradiance * 0.15 +  # Main factor: irradiance
            (30 - np.abs(temperature - 25)) * 2 +  # Optimal temperature around 25°C
            np.random.normal(0, 10, n_samples)  # Random noise
        )
        power_output = np.maximum(0, power_output)  # Ensure non-negative
        
        # Create DataFrame
        data = pd.DataFrame({
            'irradiance': irradiance,
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'hour': hour,
            'season': season,
            'power_output': power_output
        })
        
        return data
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        target_column: str = 'power_output',
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training by splitting into train/test sets and scaling.
        
        Args:
            data: Input DataFrame with features and target
            target_column: Name of the target column
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        self.feature_names = list(X.columns)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train.values, y_test.values
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using the fitted scaler.
        
        Args:
            X: Input DataFrame with features
            
        Returns:
            Scaled feature array
        """
        return self.scaler.transform(X)
