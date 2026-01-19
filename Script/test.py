import requests

url = "http://127.0.0.1:9696/predict"

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

response = requests.post(url, json=data)

# Debug
print("Status code:", response.status_code)
print("Response content:", response.text)

# Only try to parse JSON if status code is 200
if response.status_code == 200:
    print(response.json())
else:
    print("Error: API did not return valid JSON")
