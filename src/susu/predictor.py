"""
Predictor module for making solar power generation predictions.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
from .data_loader import DataLoader
from .model import SolarPowerModel


class SolarPowerPredictor:
    """
    High-level interface for solar power generation prediction.
    """
    
    def __init__(self):
        """Initialize the predictor."""
        self.data_loader = DataLoader()
        self.model = None
        self.is_trained = False
    
    def create_model(self, **xgb_params) -> None:
        """
        Create a new XGBoost model.
        
        Args:
            **xgb_params: Parameters for XGBoost model
        """
        self.model = SolarPowerModel(**xgb_params)
    
    def train(
        self,
        data: Optional[pd.DataFrame] = None,
        use_sample_data: bool = True,
        test_size: float = 0.2,
        verbose: bool = True,
        **xgb_params
    ) -> dict:
        """
        Train the prediction model.
        
        Args:
            data: Training data (if None, sample data will be generated)
            use_sample_data: Whether to use generated sample data
            test_size: Proportion of data to use for testing
            verbose: Whether to print training progress
            **xgb_params: Additional parameters for XGBoost
            
        Returns:
            Dictionary with training metrics
        """
        # Load or generate data
        if data is None and use_sample_data:
            data = self.data_loader.load_sample_data()
        elif data is None:
            raise ValueError("No data provided and use_sample_data is False")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.data_loader.prepare_data(
            data, test_size=test_size
        )
        
        # Create model if not exists
        if self.model is None:
            self.create_model(**xgb_params)
        
        # Train the model
        metrics = self.model.train(
            X_train, y_train,
            X_test, y_test,
            verbose=verbose
        )
        
        self.is_trained = True
        
        if verbose:
            print("\n=== Training Results ===")
            print(f"Train RMSE: {metrics['train_rmse']:.2f}")
            print(f"Train MAE: {metrics['train_mae']:.2f}")
            print(f"Train R²: {metrics['train_r2']:.4f}")
            if 'val_rmse' in metrics:
                print(f"Validation RMSE: {metrics['val_rmse']:.2f}")
                print(f"Validation MAE: {metrics['val_mae']:.2f}")
                print(f"Validation R²: {metrics['val_r2']:.4f}")
        
        return metrics
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions for new data.
        
        Args:
            X: Input features (array or DataFrame)
            
        Returns:
            Predicted power output values
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        # Convert DataFrame to array if needed
        if isinstance(X, pd.DataFrame):
            X = self.data_loader.transform(X)
        
        return self.model.predict(X)
    
    def get_feature_importance(self, feature_names: Optional[list] = None) -> pd.DataFrame:
        """
        Get feature importance as a DataFrame.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")
        
        importance = self.model.get_feature_importance()
        
        if feature_names is None:
            feature_names = self.data_loader.feature_names
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        # Create DataFrame and sort by importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save(self, filepath: str) -> None:
        """
        Save the trained model.
        
        Args:
            filepath: Path where to save the model
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        self.model.save_model(filepath)
    
    def load(self, filepath: str) -> None:
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        if self.model is None:
            self.create_model()
        
        self.model.load_model(filepath)
        self.is_trained = True
