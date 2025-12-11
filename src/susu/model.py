"""
XGBoost model module for solar power generation prediction.
"""

import numpy as np
import xgboost as xgb
from typing import Optional, Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class SolarPowerModel:
    """
    XGBoost-based model for predicting solar power generation.
    """
    
    def __init__(self, **xgb_params):
        """
        Initialize the solar power prediction model.
        
        Args:
            **xgb_params: Additional parameters to pass to XGBoost
        """
        # Default XGBoost parameters optimized for regression
        default_params = {
            'objective': 'reg:squarederror',  # Regression task
            'max_depth': 6,  # Maximum tree depth (balanced complexity)
            'learning_rate': 0.1,  # Step size shrinkage (conservative)
            'n_estimators': 100,  # Number of boosting rounds
            'min_child_weight': 1,  # Minimum sum of instance weight
            'subsample': 0.8,  # Fraction of samples for training
            'colsample_bytree': 0.8,  # Fraction of features for training
            'random_state': 42  # Reproducibility seed
        }
        
        # Update with user-provided parameters
        default_params.update(xgb_params)
        
        self.model = xgb.XGBRegressor(**default_params)
        self.feature_importance = None
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with training metrics
        """
        # Prepare evaluation set if validation data is provided
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        # Train the model
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=verbose
        )
        
        # Store feature importance
        self.feature_importance = self.model.feature_importances_
        
        # Calculate training metrics
        y_train_pred = self.model.predict(X_train)
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_r2': r2_score(y_train, y_train_pred)
        }
        
        # Calculate validation metrics if validation data is provided
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            metrics.update({
                'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
                'val_mae': mean_absolute_error(y_val, y_val_pred),
                'val_r2': r2_score(y_val, y_val_pred)
            })
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted power output values
        """
        return self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            X: Input features
            y: True target values
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X)
        
        return {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores.
        
        Returns:
            Array of feature importance scores
        """
        return self.feature_importance
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path where to save the model
        """
        self.model.save_model(filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from a file.
        
        Args:
            filepath: Path to the saved model
        """
        self.model.load_model(filepath)
        self.feature_importance = self.model.feature_importances_
