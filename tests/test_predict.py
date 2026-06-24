import os
import sys
import pandas as pd
import pytest

# Add parent directory to path to allow src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import load_pipeline, predict_single_customer, predict_batch_customers

def test_prediction_pipeline_loading():
    pipeline = load_pipeline()
    assert pipeline is not None, "Trained pipeline must be available. Did you run the training script?"

def test_predict_single():
    pipeline = load_pipeline()
    customer = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 1,
        'PhoneService': 'No',
        'MultipleLines': 'No phone service',
        'InternetService': 'DSL',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 29.85,
        'TotalCharges': 29.85
    }
    
    result = predict_single_customer(customer, pipeline)
    
    assert 'churn_probability' in result
    assert 'prediction' in result
    assert 'risk_level' in result
    assert isinstance(result['churn_probability'], float)
    assert result['prediction'] in [0, 1]
    assert result['risk_level'] in ['Low', 'Medium', 'High']

def test_predict_batch():
    pipeline = load_pipeline()
    customers = [
        {
            'gender': 'Female',
            'SeniorCitizen': 0,
            'Partner': 'Yes',
            'Dependents': 'No',
            'tenure': 1,
            'PhoneService': 'No',
            'MultipleLines': 'No phone service',
            'InternetService': 'DSL',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'Yes',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 29.85,
            'TotalCharges': 29.85
        },
        {
            'gender': 'Male',
            'SeniorCitizen': 0,
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': 34,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'DSL',
            'OnlineSecurity': 'Yes',
            'OnlineBackup': 'No',
            'DeviceProtection': 'Yes',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'One year',
            'PaperlessBilling': 'No',
            'PaymentMethod': 'Mailed check',
            'MonthlyCharges': 56.95,
            'TotalCharges': 1889.5
        }
    ]
    df = pd.DataFrame(customers)
    results_df = predict_batch_customers(df, pipeline)
    
    assert 'churn_probability' in results_df.columns
    assert 'prediction' in results_df.columns
    assert 'risk_level' in results_df.columns
    assert len(results_df) == 2
