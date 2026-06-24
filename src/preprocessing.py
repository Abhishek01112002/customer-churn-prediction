import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from . import config
from .utils import get_logger

logger = get_logger(__name__)

def load_and_clean_data(filepath=config.DATA_PATH):
    logger.info(f"Loading data from {filepath}")
    if not __import__('os').path.exists(filepath):
        logger.error(f"Dataset not found at {filepath}")
        raise FileNotFoundError(f"Dataset not found at {filepath}")
        
    df = pd.read_csv(filepath)
    
    logger.info("Cleaning data and handling missing values...")
    # Handle TotalCharges missing values (blank spaces)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    missing_count = df['TotalCharges'].isnull().sum()
    if missing_count > 0:
        logger.info(f"Filling {missing_count} missing values in TotalCharges with median.")
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    logger.info("Performing feature engineering...")
    # 'tenure_group': categorize tenure
    labels = ['0-12', '13-24', '25-36', '37-48', '49-60', '61-72']
    # Add a buffer up to 100 for bins to catch max values just in case
    df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 48, 60, 100], right=False, labels=labels)
    df['tenure_group'] = df['tenure_group'].astype(str) # convert from categorical to string for OHE

    # Drop customerID 
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    
    # Target encoding
    if config.TARGET in df.columns:
        df[config.TARGET] = df[config.TARGET].map({'Yes': 1, 'No': 0})
    
    return df

def get_preprocessor():
    logger.info("Building scikit-learn preprocessing pipeline...")
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
    
    # dynamically add engineered features to config for transformer
    cat_features = list(config.CATEGORICAL_FEATURES)
    if 'tenure_group' not in cat_features:
        cat_features.append('tenure_group')
        
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, config.NUMERICAL_FEATURES),
            ('cat', categorical_transformer, cat_features)
        ])
    return preprocessor
