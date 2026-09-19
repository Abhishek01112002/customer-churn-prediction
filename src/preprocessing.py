import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from . import config
from .utils import get_logger, save_object

logger = get_logger(__name__)


def load_and_clean_data(filepath=config.DATA_PATH):
    logger.info(f"Loading data from {filepath}")
    if not __import__('os').path.exists(filepath):
        logger.error(f"Dataset not found at {filepath}")
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    df = pd.read_csv(filepath)

    logger.info("Cleaning data and handling missing values...")
    # Handle TotalCharges missing values (blank spaces stored as empty strings)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    missing_count = df['TotalCharges'].isnull().sum()

    # Compute median from training data and persist it so inference uses
    # the identical value — never 0.0 or a different dataset's median.
    tc_median = float(df['TotalCharges'].median())
    if missing_count > 0:
        logger.info(f"Filling {missing_count} missing TotalCharges values with training median ({tc_median:.2f}).")
        df['TotalCharges'] = df['TotalCharges'].fillna(tc_median)

    # Persist the training-set median for use at inference time
    save_object(tc_median, config.MEDIAN_PATH)
    logger.info(f"Persisted TotalCharges training median ({tc_median:.2f}) to {config.MEDIAN_PATH}")

    logger.info("Performing feature engineering...")
    # Feature A: Service Count
    services = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['Number_of_Services'] = 0
    for col in services:
        df['Number_of_Services'] += df[col].apply(lambda x: 1 if x == 'Yes' else 0)

    # Feature B: Automatic Payment Indicator (binary 0/1 — numeric)
    df['Is_Automatic_Payment'] = df['PaymentMethod'].apply(
        lambda x: 1 if 'automatic' in str(x).lower() else 0
    )

    # Feature C: Ratio of Monthly to Total Charges
    df['Monthly_to_Total_Ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)

    # Feature D: Average monthly charges based on tenure
    df['Avg_Charges_Per_Month'] = df['TotalCharges'] / (df['tenure'] + 1)

    # Feature E: Has Internet (binary 0/1 — numeric)
    df['Has_Internet'] = df['InternetService'].apply(lambda x: 0 if x == 'No' else 1)

    # 'tenure_group': categorize tenure (used as a categorical feature)
    labels = ['0-12', '13-24', '25-36', '37-48', '49-60', '61-72']
    df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 48, 60, 100], right=False, labels=labels)
    df['tenure_group'] = df['tenure_group'].astype(str)

    # Drop customerID
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # Target encoding
    if config.TARGET in df.columns:
        df[config.TARGET] = df[config.TARGET].map({'Yes': 1, 'No': 0})

    return df


def get_preprocessor():
    logger.info("Building scikit-learn preprocessing pipeline...")
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

    # Build the categorical feature list including the engineered tenure_group
    cat_features = list(config.CATEGORICAL_FEATURES)
    if 'tenure_group' not in cat_features:
        cat_features.append('tenure_group')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, config.NUMERICAL_FEATURES),
            ('cat', categorical_transformer, cat_features)
        ])
    return preprocessor
