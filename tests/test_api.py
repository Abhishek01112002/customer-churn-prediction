import os
import sys
from fastapi.testclient import TestClient

# Add parent directory to path to allow src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.api import app

client = TestClient(app)

def test_api_read_root():
    response = client.get("/")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "online"
    assert "model_loaded" in json_data
    assert "version" in json_data

def test_api_predict_single():
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
    
    # We trigger the startup events manually inside TestClient context
    with TestClient(app) as tc:
        response = tc.post("/predict", json=customer)
        assert response.status_code == 200
        json_data = response.json()
        assert "churn_probability" in json_data
        assert "prediction" in json_data
        assert "risk_level" in json_data
        assert json_data["risk_level"] in ["Low", "Medium", "High"]

def test_api_get_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    json_data = response.json()
    assert "best_model" in json_data
    assert "best_model_metrics" in json_data
    assert "comparison" in json_data
