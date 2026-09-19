import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# We assume the user has the CSV one level up from the customer-churn directory
DATA_PATH = os.environ.get('DATA_PATH', os.path.join(BASE_DIR, '..', 'WA_Fn-UseC_-Telco-Customer-Churn.csv'))

MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Application configs
VERSION = 'v1'
MODEL_PATH = os.path.join(MODEL_DIR, f'model_{VERSION}.pkl')
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, f'preprocessor_{VERSION}.pkl')

# Path to the persisted training-set TotalCharges median used for consistent imputation
MEDIAN_PATH = os.path.join(MODEL_DIR, 'totalcharges_median.pkl')

# Path to saved pre-calibration base model (used for SHAP explanations)
BASE_MODEL_PATH = os.path.join(MODEL_DIR, f'base_model_{VERSION}.pkl')

# Feature sets
TARGET = 'Churn'

# Truly continuous / ordinal numeric features fed to StandardScaler
NUMERICAL_FEATURES = [
    'tenure', 'MonthlyCharges', 'TotalCharges',
    'Number_of_Services', 'Monthly_to_Total_Ratio', 'Avg_Charges_Per_Month',
    # Binary engineered features kept as numeric (0/1) — NOT passed to OHE
    'SeniorCitizen', 'Is_Automatic_Payment', 'Has_Internet',
]

# Purely categorical (string) features fed to OneHotEncoder
CATEGORICAL_FEATURES = [
    'gender', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod',
]

# Random state for reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Optuna hyperparameter search — number of trials
N_TRIALS = 50
