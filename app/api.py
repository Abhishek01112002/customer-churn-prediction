import os
import io
import sys
import json
import pandas as pd
from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path to allow src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import load_pipeline, predict_single_customer, predict_batch_customers
from src import config

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Production-grade API for predicting customer churn risk using machine learning models trained with SMOTE.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pipeline on startup
pipeline = None

@app.on_event("startup")
def startup_event():
    global pipeline
    pipeline = load_pipeline()

class CustomerInput(BaseModel):
    gender: str = Field(..., example="Female", description="Gender of the customer (Female, Male)")
    SeniorCitizen: int = Field(..., example=0, description="Senior citizen indicator (0, 1)")
    Partner: str = Field(..., example="Yes", description="Whether the customer has a partner (Yes, No)")
    Dependents: str = Field(..., example="No", description="Whether the customer has dependents (Yes, No)")
    tenure: int = Field(..., example=1, description="Number of months the customer has stayed with the company")
    PhoneService: str = Field(..., example="No", description="Whether the customer has a phone service (Yes, No)")
    MultipleLines: str = Field(..., example="No phone service", description="Whether the customer has multiple lines (Yes, No, No phone service)")
    InternetService: str = Field(..., example="DSL", description="Customer's internet service provider (DSL, Fiber optic, No)")
    OnlineSecurity: str = Field(..., example="No", description="Whether the customer has online security (Yes, No, No internet service)")
    OnlineBackup: str = Field(..., example="Yes", description="Whether the customer has online backup (Yes, No, No internet service)")
    DeviceProtection: str = Field(..., example="No", description="Whether the customer has device protection (Yes, No, No internet service)")
    TechSupport: str = Field(..., example="No", description="Whether the customer has tech support (Yes, No, No internet service)")
    StreamingTV: str = Field(..., example="No", description="Whether the customer has streaming TV (Yes, No, No internet service)")
    StreamingMovies: str = Field(..., example="No", description="Whether the customer has streaming movies (Yes, No, No internet service)")
    Contract: str = Field(..., example="Month-to-month", description="The contract term of the customer (Month-to-month, One year, Two year)")
    PaperlessBilling: str = Field(..., example="Yes", description="Whether the customer has paperless billing (Yes, No)")
    PaymentMethod: str = Field(..., example="Electronic check", description="The customer's payment method")
    MonthlyCharges: float = Field(..., example=29.85, description="The amount charged to the customer monthly")
    TotalCharges: float = Field(..., example=29.85, description="The total amount charged to the customer")

class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., description="Probability of customer churning (0.0 to 1.0)")
    prediction: int = Field(..., description="Binary churn prediction (1 = Churn, 0 = No Churn)")
    risk_level: str = Field(..., description="Categorized risk level (Low, Medium, High)")

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

@app.get("/", tags=["General"])
def read_root():
    """Health check endpoint showing API and model loading status."""
    return {
        "status": "online",
        "model_loaded": pipeline is not None,
        "version": config.VERSION,
        "features": {
            "numerical": config.NUMERICAL_FEATURES,
            "categorical": config.CATEGORICAL_FEATURES
        }
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_single(customer: CustomerInput):
    """Predict churn risk for a single customer profile."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded. Please train the model.")
    
    try:
        customer_dict = customer.dict()
        result = predict_single_customer(customer_dict, pipeline)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch/json", response_model=List[PredictionResponse], tags=["Inference"])
def predict_batch_json(customers: List[CustomerInput]):
    """Predict churn risk for a batch of customers provided in a JSON list."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded. Please train the model.")
    
    try:
        df = pd.DataFrame([c.dict() for c in customers])
        results_df = predict_batch_customers(df, pipeline)
        
        response = []
        for _, row in results_df.iterrows():
            response.append({
                'churn_probability': float(row['churn_probability']),
                'prediction': int(row['prediction']),
                'risk_level': row['risk_level']
            })
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.post("/predict/batch/csv", tags=["Inference"])
def predict_batch_csv(file: UploadFile = File(...)):
    """Predict churn risk for a batch of customers uploaded via a CSV file."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded. Please train the model.")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    try:
        contents = file.file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Verify required columns exist
        required_cols = config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"Uploaded CSV is missing required features: {missing_cols}"
            )
            
        results_df = predict_batch_customers(df, pipeline)
        
        # Return predictions as JSON list
        predictions = []
        for idx, row in results_df.iterrows():
            predictions.append({
                'row_index': idx,
                'churn_probability': float(row['churn_probability']),
                'prediction': int(row['prediction']),
                'risk_level': row['risk_level']
            })
            
        return {
            "total_records": len(df),
            "predictions": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV processing error: {str(e)}")

@app.get("/metrics", tags=["Metadata"])
def get_metrics():
    """Retrieve saved evaluation metrics from the last training run."""
    metrics_path = os.path.join(config.MODEL_DIR, 'metrics.json')
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Metrics file not found. Please train the model first.")
        
    try:
        with open(metrics_path, 'r') as f:
            metrics_payload = json.load(f)
        return metrics_payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading metrics: {str(e)}")
