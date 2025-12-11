# Susu - Solar Power Generation Prediction

A Python project using the **XGBoost** machine learning framework for predicting solar power generation.

## Overview

Susu is a solar power generation prediction system that uses XGBoost, a powerful gradient boosting framework, to predict power output based on various environmental factors such as:

- Solar irradiance
- Temperature
- Humidity
- Wind speed
- Time of day
- Season

## Features

- **XGBoost Integration**: Leverages the powerful XGBoost ML framework for accurate predictions
- **Data Loading & Preprocessing**: Built-in data loader with scaling and train/test splitting
- **Sample Data Generation**: Generate synthetic solar power data for testing and demonstration
- **Model Training**: Easy-to-use training interface with validation metrics
- **Feature Importance**: Analyze which features contribute most to predictions
- **Model Persistence**: Save and load trained models

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install as Package

```bash
pip install -e .
```

## Quick Start

### Basic Usage

```python
from susu import SolarPowerPredictor

# Create predictor
predictor = SolarPowerPredictor()

# Train with sample data
metrics = predictor.train(use_sample_data=True)

# Make predictions
import pandas as pd
sample_data = pd.DataFrame({
    'irradiance': [800],
    'temperature': [25],
    'humidity': [60],
    'wind_speed': [5],
    'hour': [12],
    'season': [1]
})

predictions = predictor.predict(sample_data)
print(f"Predicted power output: {predictions[0]:.2f} kW")
```

### Run Example

```bash
python examples/basic_usage.py
```

## Project Structure

```
susu/
├── src/
│   └── susu/
│       ├── __init__.py
│       ├── data_loader.py    # Data loading and preprocessing
│       ├── model.py           # XGBoost model implementation
│       └── predictor.py       # High-level prediction interface
├── tests/
│   ├── test_data_loader.py
│   ├── test_model.py
│   └── test_predictor.py
├── examples/
│   └── basic_usage.py         # Example usage script
├── requirements.txt
├── setup.py
└── README.md
```

## Usage Examples

### Training with Custom Parameters

```python
from susu import SolarPowerPredictor

predictor = SolarPowerPredictor()

# Train with custom XGBoost parameters
metrics = predictor.train(
    use_sample_data=True,
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05
)
```

### Feature Importance Analysis

```python
# Get feature importance
importance_df = predictor.get_feature_importance()
print(importance_df)
```

### Model Persistence

```python
# Save trained model
predictor.save('solar_model.json')

# Load model
new_predictor = SolarPowerPredictor()
new_predictor.load('solar_model.json')
```

## Running Tests

```bash
# Run all tests
python -m unittest discover tests

# Run specific test file
python -m unittest tests.test_model
```

## Requirements

- xgboost>=1.7.0
- numpy>=1.21.0
- pandas>=1.3.0
- scikit-learn>=1.0.0
- matplotlib>=3.4.0

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
