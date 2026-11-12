import urllib.request
import json

URL = "http://localhost:8000/predict"

registro_ejemplo = {
    "Age": 42.0,
    "Gender": "Other",
    "Country": "Saudi Arabia",
    "Height_cm": 148.5,
    "Weight_kg": 101.2,
    "BMI": 45.9,
    "Waist_Circumference_cm": 121.8,
    "Blood_Glucose": 247.3,
    "HbA1c": 11.3,
    "Fasting_Blood_Sugar": 199.3,
    "Insulin_Level": 25.4,
    "Blood_Pressure_Systolic": 145,
    "Blood_Pressure_Diastolic": 92,
    "Total_Cholesterol": 240.0,
    "HDL": 38.0,
    "LDL": 160.0,
    "Triglycerides": 210.0,
    "Heart_Rate": 82,
    "Physical_Activity_Level": "Low",
    "Exercise_Hours_Per_Week": 1.0,
    "Daily_Walking_Minutes": 15.0,
    "Diet_Quality": "Poor",
    "Sugar_Intake_Level": "High",
    "Sleep_Hours": 5.5,
    "Stress_Level": "High",
    "Smoking_Status": "Current",
    "Alcohol_Consumption": "Frequently",
    "Family_History_Diabetes": "Yes",
    "Hypertension": "Yes",
    "Heart_Disease": "Yes",
    "Fatty_Liver": "Yes",
    "PCOS": "Yes",
    "Medication_Adherence": "Poor",
    "Work_Type": "Government",
    "Residence_Type": "Rural",
    "Daily_Water_Intake_L": 2.1
}

req = urllib.request.Request(
    URL,
    data=json.dumps(registro_ejemplo).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)

try:
    with urllib.request.urlopen(req) as resp:
        print("Código HTTP:", resp.status)
        print("Respuesta:", resp.read().decode("utf-8"))
except Exception as e:
    print("Error al conectar con la API:", e)
