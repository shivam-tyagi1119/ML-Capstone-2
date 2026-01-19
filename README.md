# Loan Amount Prediction — ML Capstone Project

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-experimental-orange)](#)
[![Tests](https://img.shields.io/badge/tests-local-yellow)](#testing--visibility)

A Linear Regression-based project to predict loan amounts and demonstrate a simple ML training → serving → test loop with a Flask predict endpoint.

## Table of contents
- [Project overview](#project-overview)
- [Dataset](#dataset)
- [Features & preprocessing](#features--preprocessing)
- [Quick start](#quick-start)
- [REST API & real-time endpoint](#rest-api--real-time-endpoint)
- [Testing & visibility](#testing--visibility)
- [Example payloads & curl](#example-payloads--curl)
- [Troubleshooting](#troubleshooting)
- [Author](#author)

## Project overview
Banks need to estimate loan amounts a customer is likely to request or be eligible for based on personal and financial data. This project trains a model on historical credit-risk data and exposes a Flask endpoint to serve predictions.

## Dataset
Data sourced from: Kaggle — Credit Risk Dataset

Key fields:
- person_age — Age of customer
- person_income — Annual income
- person_emp_length — Years of employment
- person_home_ownership — Home ownership status (RENT, OWN, MORTGAGE, etc.)
- loan_int_rate — Loan interest rate
- loan_amnt — Loan amount requested (target variable)
- loan_status — Loan default status
- loan_grade — Credit grade
- cb_person_default_on_file — Default history flag

## Features & preprocessing
- Categorical features encoded with one-hot encoding
- Missing numeric values filled with median
- Boolean columns converted to integers
- Feature scaling applied using StandardScaler

## Quick start
1. Clone the repo and create a virtual environment
   ```
   git clone https://github.com/shivam-tyagi1119/ML-Capstone-2.git
   cd ML-Capstone-2
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Train the model (from the Scripts folder)
   ```
   cd Scripts
   python train.py
   ```
   This should create the saved model (check your script for the exact output filename).

3. Serve the model locally (start the Flask app)
   ```
   python predict.py
   ```
   Default endpoint: http://localhost:9696/predict (confirm in your predict.py)

4. Run the local test that hits the running endpoint
   ```
   python test.py
   ```
   test.py sends the sample payload to the running /predict endpoint and prints the response. This is the simplest way to validate the entire RTE (real-time endpoint) pipeline.

## REST API & real-time endpoint
- Endpoint: POST /predict
- Accepts: JSON payload (single object or list)
- Returns: Prediction (probability / predicted amount — follow the script's output format)

Use `flask_ping.py` to verify the server is running before running `test.py`.

## Testing & visibility
To make the repository more visible and reliable:
- Keep `test.py` simple and deterministic so it can be run in CI (it should not require external services or long-running training).
- Add a CI workflow (GitHub Actions) to run the test after pushes/PRs and publish a status badge in this README.

Sample GitHub Actions workflow you can add to `.github/workflows/ci.yml`:
```yaml
name: CI - train & test

on:
  push:
    branches: [ main, master ]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          python -m venv venv
          source venv/bin/activate
          pip install -r requirements.txt
      - name: Train model
        run: |
          cd Scripts
          python train.py
      - name: Start server in background
        run: |
          cd Scripts
          nohup python predict.py & sleep 3
      - name: Run test
        run: |
          cd Scripts
          python test.py
```

Add the Actions status badge to the top of this README (replace `OWNER/REPO` with your values):
```
[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)
```

Running CI and showing the badge significantly improves project discoverability and trustworthiness for visitors.

## Example payloads & curl
Sample JSON payload (single record):
```json
{
  "person_age": 30,
  "person_income": 50000,
  "person_emp_length": 5,
  "loan_int_rate": 12.5,
  "loan_grade_B": 0,
  "loan_grade_C": 0,
  "loan_grade_D": 1,
  "loan_grade_E": 0,
  "loan_grade_F": 0,
  "loan_grade_G": 0,
  "cb_person_default_on_file_Y": 0
}
```

Example curl:
```
curl -X POST -H "Content-Type: application/json" \
  -d '[{ "person_age": 30, "person_income": 50000, "person_emp_length": 5, "loan_int_rate": 12.5, "loan_grade_D": 1, "cb_person_default_on_file_Y": 0 }]" \
  http://localhost:9696/predict
```

Expected behavior:
- `predict.py` should return a JSON object or list containing the prediction(s).
- `test.py` should print the response and exit with code 0 on success.

## Troubleshooting
- If `test.py` fails with connection errors: ensure `predict.py` is running and reachable on the expected port.
- If model file missing after train: check `train.py` output path and verify write permissions.
- For dependency issues: verify `requirements.txt` and Python version.

## Author
Shivam Tyagi

---