Loan Amount Prediction – ML Capstone Project
Problem Statement

Banks need to estimate the loan amount a customer is likely to request or be eligible for based on their personal and financial information. Automating this process improves:

Loan approval efficiency

Customer experience

Risk management

This project uses Linear Regression to predict loan amounts for bank customers using historical credit risk data.

Dataset

The data is sourced from Kaggle – Credit Risk Dataset
.

Key Features:

Feature	Description
person_age	Age of the customer
person_income	Annual income
person_emp_length	Years of employment
person_home_ownership	Home ownership status (RENT, OWN, MORTGAGE, etc.)
loan_int_rate	Loan interest rate
loan_amnt	Loan amount requested (target variable)
loan_status	Loan default status
loan_grade	Credit grade of the customer
cb_person_default_on_file	Whether the customer has a default history

Preprocessing steps performed:

Categorical features encoded with one-hot encoding

Missing numeric values filled with median

Boolean columns converted to integers

Feature scaling applied using StandardScaler





REST API Deployment
A Flask server exposes a /predict endpoint that accepts a JSON payload and returns the predicted probability of loan default.

Steps:

Run train.py from the Script folder to generate the model.
Run predict.py from the Script folder to start the real-time Flask endpoint.
Use flask_ping.py to validate if the Flask server is running correctly.


data = [
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
]

Step 1: From the Scripts folder, run train.py to generate the model.
Step 2: From the Scripts folder, run predict.py to serve the model as a real-time endpoint (RTE).
Step 3: From the Scripts folder, run test.py to genrate the prediction for customer present in payload.

Note:
flask_ping.py is used to verify that the Flask API is running correctly.


Example curl : curl -X POST -H "Content-Type: application/json"      -d '{...}'      http://localhost:9696/predict

Author
Shivam Tyagi







