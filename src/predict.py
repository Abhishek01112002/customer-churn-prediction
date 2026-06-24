import pandas as pd
import numpy as np
from .utils import get_logger, load_object
from . import config

logger = get_logger(__name__)

def load_pipeline():
    """Loads the trained imblearn pipeline."""
    try:
        pipeline = load_object(config.MODEL_PATH)
        return pipeline
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        return None

def preprocess_inference_data(df: pd.DataFrame) -> pd.DataFrame:
    """Applies identical feature engineering to inference data as the training phase."""
    df_clean = df.copy()
    
    # Handle TotalCharges numeric conversions and missing values
    if 'TotalCharges' in df_clean.columns:
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        # We can't easily compute training median here, but we can fill with a fallback or use column values
        df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0.0)
        
    if 'tenure' in df_clean.columns:
        labels = ['0-12', '13-24', '25-36', '37-48', '49-60', '61-72']
        df_clean['tenure_group'] = pd.cut(df_clean['tenure'], bins=[0, 12, 24, 36, 48, 60, 100], right=False, labels=labels)
        df_clean['tenure_group'] = df_clean['tenure_group'].astype(str)
        
    # Drop customerID if present
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop('customerID', axis=1)
        
    return df_clean

def predict_single_customer(customer_data: dict, pipeline):
    """
    Predicts churn risk for a single customer.
    Applies identical feature engineering as the training phase.
    """
    df = pd.DataFrame([customer_data])
    df_preprocessed = preprocess_inference_data(df)
    
    try:
        proba = pipeline.predict_proba(df_preprocessed)[0][1]
        prediction = pipeline.predict(df_preprocessed)[0]
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise e
        
    risk_level = "High" if proba > 0.7 else "Medium" if proba > 0.4 else "Low"
    
    return {
        'churn_probability': float(proba),
        'prediction': int(prediction),
        'risk_level': risk_level
    }

def predict_batch_customers(df: pd.DataFrame, pipeline) -> pd.DataFrame:
    """
    Predicts churn risk for a batch of customers in a DataFrame.
    Applies preprocessing and adds prediction results.
    """
    df_preprocessed = preprocess_inference_data(df)
    
    try:
        probas = pipeline.predict_proba(df_preprocessed)[:, 1]
        predictions = pipeline.predict(df_preprocessed)
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise e
        
    results_df = df.copy()
    results_df['churn_probability'] = probas
    results_df['prediction'] = predictions
    results_df['risk_level'] = np.where(probas > 0.7, 'High', np.where(probas > 0.4, 'Medium', 'Low'))
    
    return results_df

def get_feature_importance(pipeline):
    """
    Extracts feature importances from the chosen model.
    Maps transformer output features to the model's feature_importances_ or coef_.
    """
    try:
        classifier = pipeline.named_steps['classifier']
        preprocessor = pipeline.named_steps['preprocessor']
        
        has_fi = hasattr(classifier, 'feature_importances_')
        has_coef = hasattr(classifier, 'coef_')
        
        if has_fi or has_coef:
            cat_features = list(config.CATEGORICAL_FEATURES)
            if 'tenure_group' not in cat_features:
                cat_features.append('tenure_group')
                
            ohe = preprocessor.named_transformers_['cat']
            cat_feature_names = ohe.get_feature_names_out(cat_features)
            
            all_features = np.concatenate([config.NUMERICAL_FEATURES, cat_feature_names])
            
            if has_fi:
                importances = classifier.feature_importances_
            else:
                importances = np.abs(classifier.coef_[0])
                
            # Normalize importances so they sum to 1 (optional, but nice for comparison)
            if importances.sum() > 0:
                importances = importances / importances.sum()
            
            fi_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})
            fi_df = fi_df.sort_values(by='Importance', ascending=False)
            return fi_df.head(10)
        else:
            logger.warning("Selected model does not support feature_importances_ or coef_")
            return None
    except Exception as e:
        logger.error(f"Could not extract feature importance: {e}")
        return None


