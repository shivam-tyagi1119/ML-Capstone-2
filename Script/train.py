# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

# =============================================================================
#  TRAINING Process LOGIC
# =============================================================================
def main():
    # Load data
    df = pd.read_csv("../Data/credit_risk_dataset.csv")

    # Encode categorical variables
    categorical_cols = ['loan_grade', 'cb_person_default_on_file', 'person_home_ownership', 'loan_intent']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    print(df_encoded.isna().sum())
    df_encoded = df_encoded.dropna()

    # Convert bool -> int
    bool_cols = df_encoded.select_dtypes(include='bool').columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)


    # Features & target
    X = df_encoded.drop(['loan_amnt', 'loan_status'], axis=1)
    y = df_encoded['loan_amnt']

    # Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Train Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)

    # Save model & scaler
    MODEL_VERSION = "v1.0"
    joblib.dump(lr_model, f"loan_amount_model_{MODEL_VERSION}.joblib")
    joblib.dump(scaler, f"scaler_{MODEL_VERSION}.joblib")
    joblib.dump(X_train.columns, f"features_{MODEL_VERSION}.joblib")

    print(f"Training complete. Model v{MODEL_VERSION} saved.")

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()