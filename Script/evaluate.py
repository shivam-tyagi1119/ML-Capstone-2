# evaluate.py
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# =============================================================================
# PARAMETERS
# =============================================================================
MODEL_VERSION = "v1.0"
MODEL_PATH = f"loan_amount_model_{MODEL_VERSION}.joblib"
SCALER_PATH = f"scaler_{MODEL_VERSION}.joblib"
FEATURES_PATH = f"features_{MODEL_VERSION}.joblib"
DATA_PATH = "../Data/credit_risk_dataset.csv"  # Update path if needed

# =============================================================================
# LOAD MODEL, SCALER, FEATURE ORDER
# =============================================================================
lr_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_order = joblib.load(FEATURES_PATH)

# =============================================================================
# LOAD AND PREPROCESS DATA
# =============================================================================
df = pd.read_csv(DATA_PATH)

# Encode categorical variables
categorical_cols = ['loan_grade', 'cb_person_default_on_file', 'person_home_ownership', 'loan_intent']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Convert bool -> int
bool_cols = df_encoded.select_dtypes(include='bool').columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

# Handle missing numeric values
numeric_cols = df_encoded.select_dtypes(include=['float64', 'int64']).columns
df_encoded[numeric_cols] = df_encoded[numeric_cols].fillna(df_encoded[numeric_cols].median())

# Features & target
X = df_encoded.drop(['loan_amnt', 'loan_status'], axis=1)
y = df_encoded['loan_amnt']

# Align features with training columns (fill missing one-hot columns with 0)
X = X.reindex(columns=feature_order, fill_value=0)

# =============================================================================
# SCALE FEATURES
# =============================================================================
X_scaled = scaler.transform(X)

# =============================================================================
# PREDICT
# =============================================================================
y_pred = lr_model.predict(X_scaled)

# =============================================================================
# EVALUATE
# =============================================================================
MAE = mean_absolute_error(y, y_pred)
RMSE = np.sqrt(mean_squared_error(y, y_pred))
R2 = r2_score(y, y_pred)

print(f"Model v{MODEL_VERSION} Evaluation Metrics:")
print(f"MAE  : {MAE:.2f}")
print(f"RMSE : {RMSE:.2f}")
print(f"R2   : {R2:.3f}")
