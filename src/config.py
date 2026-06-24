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

# Feature sets
TARGET = 'Churn'
NUMERICAL_FEATURES = [
    'tenure', 'MonthlyCharges', 'TotalCharges', 
    'Number_of_Services', 'Monthly_to_Total_Ratio', 'Avg_Charges_Per_Month'
]
CATEGORICAL_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod',
    'Is_Automatic_Payment', 'Has_Internet'
]

# Random state for reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.2
