from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model, scaler, feature order
MODEL_VERSION = "v1.0"
lr_model = joblib.load(f"loan_amount_model_{MODEL_VERSION}.joblib")
scaler = joblib.load(f"scaler_{MODEL_VERSION}.joblib")
feature_order = joblib.load(f"features_{MODEL_VERSION}.joblib")

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        data = request.get_json()  # Get JSON input
        df = pd.DataFrame(data)
        
        # Compute any derived features (if needed)
        if 'income_per_age' in feature_order and 'income_per_age' not in df.columns:
            df['income_per_age'] = df['person_income'] / df['person_age']

        # Align features to training
        df = df.reindex(columns=feature_order, fill_value=0)

        # Scale
        X_scaled = scaler.transform(df)

        # Predict
        y_pred = lr_model.predict(X_scaled)

        return jsonify({"predictions": y_pred.tolist()})

    except Exception as e:
        # Always return JSON even if error occurs
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696)
