import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
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

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics = {
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall': round(recall_score(y_test, y_pred), 4),
        'F1': round(f1_score(y_test, y_pred), 4)
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
    plt.title(f'Confusion Matrix - {best_model_name}')
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
        plt.title(f'Receiver Operating Characteristic (ROC) - {best_model_name}')
        plt.legend(loc="lower right")
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
        plt.title(f'Top 10 Feature Importances - {best_model_name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        fi_path = os.path.join(config.MODEL_DIR, 'feature_importance.png')
        plt.savefig(fi_path, dpi=150)
        plt.close()
        logger.info(f"Saved Feature Importance Plot to {fi_path}")

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
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=config.RANDOM_STATE, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=config.RANDOM_STATE, max_depth=8),
        'Random Forest': RandomForestClassifier(random_state=config.RANDOM_STATE, n_estimators=100),
        'XGBoost': XGBClassifier(random_state=config.RANDOM_STATE, use_label_encoder=False, eval_metric='logloss')
    }
    
    results = []
    best_model_name = None
    best_f1 = 0
    best_pipeline = None

    logger.info("Starting Cross-Validation and Evaluation for multiple models...")
    for name, model in models.items():
        # ImbPipeline correctly routes data and only applies SMOTE during fit!
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', smote),
            ('classifier', model)
        ])
        
        # Cross Validation Evaluation on F1 Score
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
        logger.info(f"{name} CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Final train and evaluation on held-out test set
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        metrics['Model'] = name
        results.append(metrics)
        
        if metrics['F1'] > best_f1:
            best_f1 = metrics['F1']
            best_model_name = name
            best_pipeline = pipeline

    results_df = pd.DataFrame(results).set_index('Model')
    
    print("\n" + "="*50)
    print("Model Comparison Table")
    print("="*50)
    print(results_df.to_markdown())
    print("="*50)
    
    logger.info(f"Best model based on F1 Score: {best_model_name}")
    logger.info(f"Saving {best_model_name} pipeline (Preproc -> SMOTE -> Model) to models/")
    
    save_object(best_pipeline, config.MODEL_PATH)
    
    # Save plots for the best model
    save_evaluation_plots(best_pipeline, X_test, y_test, best_model_name)
    
    # Save all metrics to a JSON file
    metrics_path = os.path.join(config.MODEL_DIR, 'metrics.json')
    best_metrics = results_df.loc[best_model_name].to_dict()
    metrics_payload = {
        'best_model': best_model_name,
        'best_model_metrics': best_metrics,
        'comparison': results
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_payload, f, indent=4)
    logger.info(f"Saved model metrics to {metrics_path}")

if __name__ == "__main__":
    run_training()

