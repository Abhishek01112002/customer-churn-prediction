import os
import io
import sys
import json
import pandas as pd
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN

# Add parent directory to path to allow src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import (
    load_pipeline,
    load_base_pipeline,
    predict_single_customer,
    predict_batch_customers,
    explain_single_customer,
)
from src import config

from contextlib import asynccontextmanager

# ---------------------------------------------------------------------------
# Pipeline loading & Lifespan
# ---------------------------------------------------------------------------

pipeline = None       # Calibrated — used for predictions
base_pipeline = None  # Pre-calibration — used for SHAP


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, base_pipeline
    pipeline = load_pipeline()
    base_pipeline = load_base_pipeline()
    yield


# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description=(
        "Production-grade API for predicting customer churn risk. "
        "Protected prediction and explanation endpoints require an X-API-Key header."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — whitelist specific origins via ALLOWED_ORIGINS env var
# ---------------------------------------------------------------------------

_raw_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:8501")
ALLOWED_ORIGINS: List[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# API Key authentication
# ---------------------------------------------------------------------------

_API_KEY = os.environ.get("API_KEY", "changeme")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: Optional[str] = Security(_api_key_header)):
    """
    FastAPI dependency that enforces API key authentication.
    - 401 if the header is missing.
    - 403 if the key is wrong.
    Public routes (health check, /metrics) are excluded from this dependency.
    """
    if not api_key:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header.",
        )
    if api_key != _API_KEY:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )
    return api_key


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class CustomerInput(BaseModel):
    gender: str = Field(..., examples=["Female"], description="Gender of the customer (Female, Male)")
    SeniorCitizen: int = Field(..., examples=[0], description="Senior citizen indicator (0, 1)")
    Partner: str = Field(..., examples=["Yes"], description="Whether the customer has a partner (Yes, No)")
    Dependents: str = Field(..., examples=["No"], description="Whether the customer has dependents (Yes, No)")
    tenure: int = Field(..., examples=[1], description="Number of months the customer has stayed with the company")
    PhoneService: str = Field(..., examples=["No"], description="Whether the customer has a phone service (Yes, No)")
    MultipleLines: str = Field(..., examples=["No phone service"], description="Whether the customer has multiple lines")
    InternetService: str = Field(..., examples=["DSL"], description="Customer's internet service provider (DSL, Fiber optic, No)")
    OnlineSecurity: str = Field(..., examples=["No"], description="Whether the customer has online security")
    OnlineBackup: str = Field(..., examples=["Yes"], description="Whether the customer has online backup")
    DeviceProtection: str = Field(..., examples=["No"], description="Whether the customer has device protection")
    TechSupport: str = Field(..., examples=["No"], description="Whether the customer has tech support")
    StreamingTV: str = Field(..., examples=["No"], description="Whether the customer has streaming TV")
    StreamingMovies: str = Field(..., examples=["No"], description="Whether the customer has streaming movies")
    Contract: str = Field(..., examples=["Month-to-month"], description="The contract term of the customer")
    PaperlessBilling: str = Field(..., examples=["Yes"], description="Whether the customer has paperless billing (Yes, No)")
    PaymentMethod: str = Field(..., examples=["Electronic check"], description="The customer's payment method")
    MonthlyCharges: float = Field(..., examples=[29.85], description="The amount charged to the customer monthly")
    TotalCharges: float = Field(..., examples=[29.85], description="The total amount charged to the customer")


class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., description="Probability of customer churning (0.0 to 1.0)")
    prediction: int = Field(..., description="Binary churn prediction (1 = Churn, 0 = No Churn)")
    risk_level: str = Field(..., description="Categorized risk level (Low, Medium, High)")


class ExplainResponse(BaseModel):
    churn_probability: float
    prediction: int
    risk_level: str
    shap_values: dict = Field(..., description="Top feature SHAP contributions {feature: shap_value}")
    base_value: float = Field(..., description="Expected model output (SHAP base value)")
    prediction_score: float = Field(..., description="sum(shap_values) + base_value")


# ---------------------------------------------------------------------------
# Public endpoints (no auth required)
# ---------------------------------------------------------------------------

@app.get("/", tags=["General"])
def read_root():
    """Health check endpoint showing API and model loading status."""
    return {
        "status": "online",
        "model_loaded": pipeline is not None,
        "version": config.VERSION,
        "allowed_origins": ALLOWED_ORIGINS,
        "features": {
            "numerical": config.NUMERICAL_FEATURES,
            "categorical": config.CATEGORICAL_FEATURES,
        },
    }


@app.get("/metrics", tags=["Metadata"])
def get_metrics():
    """Retrieve saved evaluation metrics from the last training run."""
    metrics_path = os.path.join(config.MODEL_DIR, 'metrics.json')
    if not os.path.exists(metrics_path):
        raise HTTPException(
            status_code=404,
            detail="Metrics file not found. Please train the model first.",
        )
    try:
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading metrics: {str(e)}")


# ---------------------------------------------------------------------------
# Protected endpoints (X-API-Key required)
# ---------------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    dependencies=[Depends(require_api_key)],
)
def predict_single(customer: CustomerInput):
    """Predict churn risk for a single customer profile."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded. Please train the model.")
    try:
        return predict_single_customer(customer.dict(), pipeline)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post(
    "/predict/batch/json",
    response_model=List[PredictionResponse],
    tags=["Inference"],
    dependencies=[Depends(require_api_key)],
)
def predict_batch_json(customers: List[CustomerInput]):
    """Predict churn risk for a batch of customers provided in a JSON list."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded. Please train the model.")
    try:
        df = pd.DataFrame([c.dict() for c in customers])
        results_df = predict_batch_customers(df, pipeline)
        return [
            {
                'churn_probability': float(row['churn_probability']),
                'prediction': int(row['prediction']),
                'risk_level': row['risk_level'],
            }
            for _, row in results_df.iterrows()
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.post(
    "/predict/batch/csv",
    tags=["Inference"],
    dependencies=[Depends(require_api_key)],
)
def predict_batch_csv(file: UploadFile = File(...)):
    """Predict churn risk for a batch of customers uploaded via a CSV file."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded. Please train the model.")
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    try:
        contents = file.file.read()
        df = pd.read_csv(io.BytesIO(contents))

        required_cols = config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Uploaded CSV is missing required features: {missing_cols}",
            )

        results_df = predict_batch_customers(df, pipeline)
        predictions = [
            {
                'row_index': idx,
                'churn_probability': float(row['churn_probability']),
                'prediction': int(row['prediction']),
                'risk_level': row['risk_level'],
            }
            for idx, row in results_df.iterrows()
        ]
        return {"total_records": len(df), "predictions": predictions}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV processing error: {str(e)}")


@app.post(
    "/explain",
    response_model=ExplainResponse,
    tags=["Explainability"],
    dependencies=[Depends(require_api_key)],
)
def explain_single(customer: CustomerInput):
    """
    Return a prediction plus SHAP feature contributions for a single customer.

    Uses the pre-calibration base model for fast TreeExplainer / LinearExplainer
    explanations. The churn_probability returned here is from the calibrated model.
    """
    if pipeline is None or base_pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded. Please train the model.")
    try:
        customer_dict = customer.dict()

        # Calibrated probability for the prediction
        pred_result = predict_single_customer(customer_dict, pipeline)

        # SHAP values from the base (pre-calibration) pipeline
        shap_result = explain_single_customer(customer_dict, base_pipeline, top_n=10)

        return {
            **pred_result,
            **shap_result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")
