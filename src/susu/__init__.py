"""
Susu - A solar power generation prediction project using XGBoost ML framework.
"""

__version__ = "0.1.0"

from .data_loader import DataLoader
from .model import SolarPowerModel
from .predictor import SolarPowerPredictor

__all__ = ["DataLoader", "SolarPowerModel", "SolarPowerPredictor"]
