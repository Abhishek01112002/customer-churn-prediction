"""
src/tune.py
-----------
Optuna hyperparameter search for the Customer Churn Prediction model.

Usage:
    python -m src.tune                   # runs N_TRIALS trials (default 50)
    python -m src.tune --trials 100      # override trial count

After completion the best parameters are saved to models/best_params.json.
The next call to `python -m src.train` will automatically pick them up.
"""

import argparse
import json
import os
import warnings

import numpy as np
import optuna
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from . import config
from .preprocessing import load_and_clean_data, get_preprocessor
from .utils import get_logger

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def _objective(trial: optuna.Trial, X_train, y_train) -> float:
    """
    Optuna objective: 5-fold stratified CV, optimising mean F1 score.
    Searches across three model families and their key hyperparameters.
    """
    family = trial.suggest_categorical(
        'model_family', ['XGBoost', 'RandomForest', 'LogisticRegression']
    )

    if family == 'XGBoost':
        clf = XGBClassifier(
            random_state=config.RANDOM_STATE,
            eval_metric='logloss',
            n_estimators=trial.suggest_int('n_estimators', 50, 500, step=50),
            max_depth=trial.suggest_int('max_depth', 2, 8),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            subsample=trial.suggest_float('subsample', 0.6, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
            scale_pos_weight=trial.suggest_float('scale_pos_weight', 1.0, 5.0),
        )

    elif family == 'RandomForest':
        clf = RandomForestClassifier(
            random_state=config.RANDOM_STATE,
            class_weight='balanced',
            n_estimators=trial.suggest_int('n_estimators', 50, 400, step=50),
            max_depth=trial.suggest_int('max_depth', 3, 15),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
            max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        )

    else:  # LogisticRegression
        clf = LogisticRegression(
            random_state=config.RANDOM_STATE,
            max_iter=1000,
            class_weight='balanced',
            C=trial.suggest_float('C', 1e-3, 10.0, log=True),
            solver=trial.suggest_categorical('solver', ['lbfgs', 'saga']),
        )

    preprocessor = get_preprocessor()
    smote = SMOTE(random_state=config.RANDOM_STATE)

    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', smote),
        ('classifier', clf),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    return float(scores.mean())


# ---------------------------------------------------------------------------
# Main tuning entry point
# ---------------------------------------------------------------------------

def run_tuning(n_trials: int = config.N_TRIALS):
    logger.info(f"Starting Optuna hyperparameter search ({n_trials} trials)...")

    df = load_and_clean_data()
    X = df.drop(config.TARGET, axis=1)
    y = df[config.TARGET]

    # Use 80 % of data for tuning (mirrors the train split in train.py)
    from sklearn.model_selection import train_test_split
    X_train, _, y_train, _ = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    study = optuna.create_study(
        direction='maximize',
        study_name='churn_prediction',
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    )

    study.optimize(
        lambda trial: _objective(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_trial = study.best_trial
    best_family = best_trial.params['model_family']
    # Params specific to the best family (exclude the model_family key itself)
    best_model_params = {k: v for k, v in best_trial.params.items() if k != 'model_family'}

    result = {
        'model_family': best_family,
        'params': best_model_params,
        'best_cv_f1': round(best_trial.value, 4),
        'n_trials': n_trials,
    }

    output_path = os.path.join(config.MODEL_DIR, 'best_params.json')
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)

    print("\n" + "=" * 55)
    print("Optuna Tuning Complete")
    print("=" * 55)
    print(f"  Best model family : {best_family}")
    print(f"  Best CV F1 score  : {best_trial.value:.4f}")
    print(f"  Best params       : {best_model_params}")
    print(f"  Saved to          : {output_path}")
    print("=" * 55)
    print("\nRun 'python -m src.train' to train with the best params.\n")

    logger.info(f"Best params saved to {output_path}")
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for churn model.")
    parser.add_argument(
        '--trials', type=int, default=config.N_TRIALS,
        help=f"Number of Optuna trials (default: {config.N_TRIALS})"
    )
    args = parser.parse_args()
    run_tuning(n_trials=args.trials)
