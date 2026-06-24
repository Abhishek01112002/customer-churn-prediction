import os
import sys
import pandas as pd
import numpy as np
import pytest

# Add parent directory to path to allow src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import preprocess_inference_data
from src import config

def test_preprocess_inference_data_basic():
    # Construct mock raw dataframe matching customer features
    data = {
        'customerID': ['1234-ABCD'],
        'gender': ['Female'],
        'SeniorCitizen': [0],
        'Partner': ['Yes'],
        'Dependents': ['No'],
        'tenure': [5],
        'PhoneService': ['No'],
        'MultipleLines': ['No phone service'],
        'InternetService': ['DSL'],
        'OnlineSecurity': ['No'],
        'OnlineBackup': ['Yes'],
        'DeviceProtection': ['No'],
        'TechSupport': ['No'],
        'StreamingTV': ['No'],
        'StreamingMovies': ['No'],
        'Contract': ['Month-to-month'],
        'PaperlessBilling': ['Yes'],
        'PaymentMethod': ['Electronic check'],
        'MonthlyCharges': [29.85],
        'TotalCharges': ['29.85'] # String to check conversion
    }
    df = pd.DataFrame(data)
    
    # Run preprocessing function (from src.predict)
    from src.predict import preprocess_inference_data
    df_clean = preprocess_inference_data(df)
    
    # Check that customerID was dropped
    assert 'customerID' not in df_clean.columns
    
    # Check that TotalCharges is float
    assert df_clean['TotalCharges'].dtype == np.float64
    assert df_clean['TotalCharges'].iloc[0] == 29.85
    
    # Check that tenure_group was added
    assert 'tenure_group' in df_clean.columns
    assert df_clean['tenure_group'].iloc[0] == '0-12'

def test_preprocess_inference_data_missing_total_charges():
    data = {
        'customerID': ['5678-EFGH'],
        'gender': ['Male'],
        'SeniorCitizen': [1],
        'Partner': ['No'],
        'Dependents': ['No'],
        'tenure': [25],
        'PhoneService': ['Yes'],
        'MultipleLines': ['No'],
        'InternetService': ['Fiber optic'],
        'OnlineSecurity': ['No'],
        'OnlineBackup': ['No'],
        'DeviceProtection': ['No'],
        'TechSupport': ['No'],
        'StreamingTV': ['No'],
        'StreamingMovies': ['No'],
        'Contract': ['Month-to-month'],
        'PaperlessBilling': ['Yes'],
        'PaymentMethod': ['Mailed check'],
        'MonthlyCharges': [70.0],
        'TotalCharges': [' ']  # Empty space representation of missing value
    }
    df = pd.DataFrame(data)
    
    from src.predict import preprocess_inference_data
    df_clean = preprocess_inference_data(df)
    
    # Check that empty TotalCharges was handled and filled
    assert df_clean['TotalCharges'].iloc[0] == 0.0
    # Check tenure group category
    assert df_clean['tenure_group'].iloc[0] == '25-36'
