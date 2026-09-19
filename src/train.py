import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

from . import config
from .preprocessing import load_and_clean_data, get_preprocessor
from .utils import get_logger, save_object
from .predict import get_feature_importance

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_best_params() -> dict:
    """
    Loads Optuna best params from models/best_params.json if it exists.
    Returns an empty dict if the file is absent (first-run or tuning skipped).
    """
    best_params_path = os.path.join(config.MODEL_DIR, 'best_params.json')
    if os.path.exists(best_params_path):
        with open(best_params_path, 'r') as f:
            params = json.load(f)
        logger.info(f"Loaded Optuna best params from {best_params_path}: {params}")
        return params
    logger.info("No best_params.json found — using default hyperparameters.")
    return {}


def _build_models(best_params: dict) -> dict:
    """
    Constructs model instances, injecting Optuna-tuned params when available.
    Falls back to sensible defaults if no params are present.
    """
    family = best_params.get('model_family', None)
    p = best_params.get('params', {})

    if family == 'XGBoost':
        tuned_xgb = XGBClassifier(
            random_state=config.RANDOM_STATE,
            eval_metric='logloss',
            n_estimators=p.get('n_estimators', 150),
            max_depth=p.get('max_depth', 4),
            learning_rate=p.get('learning_rate', 0.05),
            subsample=p.get('subsample', 0.8),
            colsample_bytree=p.get('colsample_bytree', 0.8),
            scale_pos_weight=p.get('scale_pos_weight', 1.5),
        )
        return {'XGBoost (tuned)': tuned_xgb}

    if family == 'RandomForest':
        tuned_rf = RandomForestClassifier(
            random_state=config.RANDOM_STATE,
            class_weight='balanced',
            n_estimators=p.get('n_estimators', 200),
            max_depth=p.get('max_depth', 10),
            min_samples_split=p.get('min_samples_split', 5),
            min_samples_leaf=p.get('min_samples_leaf', 2),
        )
        return {'RandomForest (tuned)': tuned_rf}

    if family == 'LogisticRegression':
        tuned_lr = LogisticRegression(
            random_state=config.RANDOM_STATE,
            max_iter=1000,
            class_weight='balanced',
            C=p.get('C', 1.0),
            solver=p.get('solver', 'lbfgs'),
        )
        return {'LogisticRegression (tuned)': tuned_lr}

    # No tuned params — run full model comparison
    return {
        'Logistic Regression': LogisticRegression(
            random_state=config.RANDOM_STATE, max_iter=1000, class_weight='balanced'
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=config.RANDOM_STATE, max_depth=8
        ),
        'Random Forest': RandomForestClassifier(
            random_state=config.RANDOM_STATE, n_estimators=200, max_depth=10,
            min_samples_split=5, class_weight='balanced'
        ),
        'XGBoost': XGBClassifier(
            random_state=config.RANDOM_STATE, n_estimators=150, max_depth=4,
            learning_rate=0.05, scale_pos_weight=1.5, eval_metric='logloss'
        ),
    }


# ---------------------------------------------------------------------------
# Evaluation / plotting
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    metrics = {
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall': round(recall_score(y_test, y_pred), 4),
        'F1': round(f1_score(y_test, y_pred), 4),
    }
    if y_proba is not None:
        metrics['ROC-AUC'] = round(roc_auc_score(y_test, y_proba), 4)

    return metrics


def save_evaluation_plots(pipeline, X_test, y_test, best_model_name):
    logger.info("Generating and saving evaluation plots...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, 'predict_proba') else None

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.title(f'Confusion Matrix — {best_model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    cm_path = os.path.join(config.MODEL_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    logger.info(f"Saved Confusion Matrix to {cm_path}")

    # 2. ROC Curve
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) — {best_model_name}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        roc_path = os.path.join(config.MODEL_DIR, 'roc_curve.png')
        plt.savefig(roc_path, dpi=150)
        plt.close()
        logger.info(f"Saved ROC Curve to {roc_path}")

    # 3. Feature Importance
    fi_df = get_feature_importance(pipeline)
    if fi_df is not None:
        plt.figure(figsize=(8, 5))
        sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis')
        plt.title(f'Top 10 Feature Importances — {best_model_name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        fi_path = os.path.join(config.MODEL_DIR, 'feature_importance.png')
        plt.savefig(fi_path, dpi=150)
        plt.close()
        logger.info(f"Saved Feature Importance Plot to {fi_path}")


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def run_training():
    logger.info("Starting model training pipeline...")
    df = load_and_clean_data()
    X = df.drop(config.TARGET, axis=1)
    y = df[config.TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    logger.info("Applying preprocessor and SMOTE for class imbalance handling...")

    preprocessor = get_preprocessor()
    smote = SMOTE(random_state=config.RANDOM_STATE)

    best_params = _load_best_params()
    models = _build_models(best_params)

    results = []
    best_model_name = None
    best_f1 = 0
    best_base_pipeline = None   # Pre-calibration pipeline (used for SHAP)

    logger.info("Starting Cross-Validation and Evaluation for model(s)...")
    for name, model in models.items():
        # ImbPipeline correctly routes data and only applies SMOTE during fit
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', smote),
            ('classifier', model),
        ])

        # 5-fold CV on F1
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
        logger.info(f"{name} CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Final fit and evaluation
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        metrics['Model'] = name
        results.append(metrics)

        if metrics['F1'] > best_f1:
            best_f1 = metrics['F1']
            best_model_name = name
            best_base_pipeline = pipeline

    results_df = pd.DataFrame(results).set_index('Model')

    print("\n" + "=" * 50)
    print("Model Comparison Table")
    print("=" * 50)
    print(results_df.to_markdown())
    print("=" * 50)

    logger.info(f"Best model based on F1: {best_model_name}")

    # ── Save the pre-calibration base pipeline for SHAP explanations ──
    logger.info(f"Saving base (pre-calibration) pipeline to {config.BASE_MODEL_PATH}")
    save_object(best_base_pipeline, config.BASE_MODEL_PATH)

    # ── Probability calibration with Platt scaling (5-fold) ──
    # We calibrate only the classifier step, keeping the preprocessor intact.
    logger.info("Applying Platt probability calibration (CalibratedClassifierCV, cv=5)...")
    base_clf = best_base_pipeline.named_steps['classifier']
    calibrated_clf = CalibratedClassifierCV(base_clf, method='sigmoid', cv=5)

    # Build a new pipeline with the calibrated classifier
    calibrated_pipeline = ImbPipeline(steps=[
        ('preprocessor', best_base_pipeline.named_steps['preprocessor']),
        ('smote', best_base_pipeline.named_steps['smote']),
        ('classifier', calibrated_clf),
    ])
    calibrated_pipeline.fit(X_train, y_train)

    cal_metrics = evaluate_model(calibrated_pipeline, X_test, y_test)
    logger.info(f"Calibrated model metrics: {cal_metrics}")

    # ── Save the calibrated pipeline as the primary model ──
    logger.info(f"Saving calibrated pipeline to {config.MODEL_PATH}")
    save_object(calibrated_pipeline, config.MODEL_PATH)

    # Save plots for the calibrated model
    save_evaluation_plots(calibrated_pipeline, X_test, y_test, best_model_name)

    # Save all metrics to a JSON file
    metrics_path = os.path.join(config.MODEL_DIR, 'metrics.json')
    best_metrics = results_df.loc[best_model_name].to_dict()
    best_metrics.update({k: round(v, 4) for k, v in cal_metrics.items() if k != 'Model'})
    metrics_payload = {
        'best_model': best_model_name,
        'best_model_metrics': best_metrics,
        'calibrated': True,
        'comparison': results,
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_payload, f, indent=4)
    logger.info(f"Saved model metrics to {metrics_path}")


if __name__ == "__main__":
    run_training()
