import os
import pandas as pd
import numpy as np
from .utils import get_logger, load_object
from . import config

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pipeline / median loading
# ---------------------------------------------------------------------------

def load_pipeline():
    """Loads the trained calibrated imblearn pipeline."""
    try:
        pipeline = load_object(config.MODEL_PATH)
        return pipeline
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        return None


def load_base_pipeline():
    """Loads the pre-calibration base model pipeline (used for SHAP)."""
    try:
        pipeline = load_object(config.BASE_MODEL_PATH)
        return pipeline
    except Exception as e:
        logger.warning(f"Base model not found, falling back to main pipeline for SHAP: {e}")
        return load_pipeline()


def _load_tc_median() -> float:
    """Returns the training-set TotalCharges median persisted during training."""
    if os.path.exists(config.MEDIAN_PATH):
        try:
            return float(load_object(config.MEDIAN_PATH))
        except Exception as e:
            logger.warning(f"Could not load TotalCharges median ({e}). Falling back to 0.0")
    return 0.0


# ---------------------------------------------------------------------------
# Inference preprocessing
# ---------------------------------------------------------------------------

def preprocess_inference_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies identical feature engineering to inference data as the training phase.
    Uses the persisted training-set TotalCharges median for NaN imputation so the
    model sees the same distribution it was trained on.
    """
    df_clean = df.copy()

    # Handle TotalCharges: convert then fill with training median (not 0.0)
    if 'TotalCharges' in df_clean.columns:
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        tc_median = _load_tc_median()
        df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(tc_median)

    # Feature A: Service Count
    services = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df_clean['Number_of_Services'] = 0
    for col in services:
        if col in df_clean.columns:
            df_clean['Number_of_Services'] += df_clean[col].apply(lambda x: 1 if x == 'Yes' else 0)

    # Feature B: Automatic Payment Indicator
    if 'PaymentMethod' in df_clean.columns:
        df_clean['Is_Automatic_Payment'] = df_clean['PaymentMethod'].apply(
            lambda x: 1 if 'automatic' in str(x).lower() else 0
        )
    else:
        df_clean['Is_Automatic_Payment'] = 0

    # Feature C: Ratio of Monthly to Total Charges
    if 'MonthlyCharges' in df_clean.columns and 'TotalCharges' in df_clean.columns:
        df_clean['Monthly_to_Total_Ratio'] = df_clean['MonthlyCharges'] / (df_clean['TotalCharges'] + 1)
    else:
        df_clean['Monthly_to_Total_Ratio'] = 0.0

    # Feature D: Average monthly charges based on tenure
    if 'TotalCharges' in df_clean.columns and 'tenure' in df_clean.columns:
        df_clean['Avg_Charges_Per_Month'] = df_clean['TotalCharges'] / (df_clean['tenure'] + 1)
    else:
        df_clean['Avg_Charges_Per_Month'] = 0.0

    # Feature E: Has Internet
    if 'InternetService' in df_clean.columns:
        df_clean['Has_Internet'] = df_clean['InternetService'].apply(lambda x: 0 if x == 'No' else 1)
    else:
        df_clean['Has_Internet'] = 0

    # Tenure group
    if 'tenure' in df_clean.columns:
        labels = ['0-12', '13-24', '25-36', '37-48', '49-60', '61-72']
        df_clean['tenure_group'] = pd.cut(
            df_clean['tenure'], bins=[0, 12, 24, 36, 48, 60, 100], right=False, labels=labels
        )
        df_clean['tenure_group'] = df_clean['tenure_group'].astype(str)

    # Drop customerID if present
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop('customerID', axis=1)

    return df_clean


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def get_feature_importance(pipeline):
    """
    Extracts feature importances from the chosen model.
    Maps transformer output features to the model's feature_importances_ or coef_.
    Works with both calibrated and uncalibrated pipelines.
    """
    try:
        # CalibratedClassifierCV wraps the base estimator — unwrap if needed
        classifier = pipeline.named_steps['classifier']
        if hasattr(classifier, 'calibrated_classifiers_'):
            # Get the base estimator from the first calibrated fold
            classifier = classifier.calibrated_classifiers_[0].estimator

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

            # Normalize so importances sum to 1
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


# ---------------------------------------------------------------------------
# SHAP Explainability
# ---------------------------------------------------------------------------

def explain_single_customer(customer_data: dict, base_pipeline, top_n: int = 10) -> dict:
    """
    Returns SHAP-based feature contributions for a single customer.

    Uses the pre-calibration base pipeline (TreeExplainer for tree models,
    LinearExplainer for linear models) for fast, accurate explanations.

    Args:
        customer_data: Raw customer feature dict (same format as CustomerInput).
        base_pipeline:  The pre-calibration imblearn pipeline (preprocessor + classifier).
        top_n: Number of top features to return.

    Returns:
        dict with keys:
          - 'shap_values': {feature_name: shap_value} sorted by |shap_value| descending
          - 'base_value': expected model output (log-odds for tree models)
          - 'prediction_score': sum(shap_values) + base_value (before sigmoid for trees)
    """
    try:
        import shap
    except ImportError:
        raise ImportError("shap is required for explanations. Install it via: pip install shap")

    df = pd.DataFrame([customer_data])
    df_preprocessed = preprocess_inference_data(df)

    # Transform input through the preprocessor step only
    preprocessor = base_pipeline.named_steps['preprocessor']
    # Ensure X_transformed is a dense 2D float array
    if hasattr(X_transformed, 'toarray'):
        X_dense = X_transformed.toarray().astype(float)
    else:
        X_dense = np.asarray(X_transformed, dtype=float)

    # Retrieve the classifier (unwrap CalibratedClassifierCV if needed)
    classifier = base_pipeline.named_steps['classifier']
    if hasattr(classifier, 'calibrated_classifiers_'):
        classifier = classifier.calibrated_classifiers_[0].estimator

    # Build feature names for the transformed matrix
    cat_features = list(config.CATEGORICAL_FEATURES)
    if 'tenure_group' not in cat_features:
        cat_features.append('tenure_group')
    ohe = preprocessor.named_transformers_['cat']
    cat_feature_names = list(ohe.get_feature_names_out(cat_features))
    all_feature_names = list(config.NUMERICAL_FEATURES) + cat_feature_names

    # Choose the fastest compatible explainer
    if hasattr(classifier, 'feature_importances_'):
        # Tree-based models (XGBoost, RandomForest, DecisionTree)
        explainer = shap.TreeExplainer(classifier)
        raw_shap = explainer.shap_values(X_dense)
        exp_val = getattr(explainer, 'expected_value', 0.0)
    elif hasattr(classifier, 'coef_'):
        # Linear models (LogisticRegression)
        explainer = shap.LinearExplainer(classifier, X_dense)
        raw_shap = explainer.shap_values(X_dense)
        exp_val = getattr(explainer, 'expected_value', 0.0)
    else:
        raise ValueError(f"No compatible SHAP explainer for classifier type: {type(classifier)}")

    # Extract base value as a single scalar float
    if isinstance(exp_val, (list, tuple, np.ndarray)):
        base_value = float(exp_val[1]) if len(exp_val) > 1 else float(exp_val[0])
    else:
        base_value = float(exp_val)

    # Extract class 1 (churn / positive class) attributions
    if isinstance(raw_shap, list):
        # List of arrays [class 0, class 1]
        shap_arr = raw_shap[1] if len(raw_shap) > 1 else raw_shap[0]
    elif hasattr(raw_shap, 'values'):
        # shap.Explanation object
        shap_arr = raw_shap.values
    else:
        shap_arr = np.asarray(raw_shap)

    # If 3D array (samples, features, classes), slice out class 1
    if shap_arr.ndim == 3:
        shap_arr = shap_arr[:, :, 1] if shap_arr.shape[2] > 1 else shap_arr[:, :, 0]

    # Flatten the row for the single customer record
    shap_row = np.asarray(shap_arr)[0].flatten()

    # Pair each feature with its attribution scalar float
    raw_pairs = sorted(
        zip(all_feature_names, [float(v) for v in shap_row]),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    top_pairs = raw_pairs[:top_n]
    shap_dict = {feat: round(float(val), 6) for feat, val in top_pairs}
    prediction_score = round(float(np.sum(shap_row)) + base_value, 6)

    return {
        'shap_values': shap_dict,
        'base_value': round(base_value, 6),
        'prediction_score': prediction_score,
    }
