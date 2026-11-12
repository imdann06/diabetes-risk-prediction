import os
import sys
from pathlib import Path
from typing import Optional
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Permitir importar módulos de fase-2
BASE_DIR = Path(__file__).resolve().parent.parent
FASE2_DIR = BASE_DIR / "fase-2"
sys.path.append(str(FASE2_DIR))

from train import build_parser, run_training, TrainingError

app = FastAPI(title="API Riesgo de Diabetes - Fase 3", version="1.0.0")

MODELS_DIR = Path(os.getenv("MODELS_DIR", FASE2_DIR / "models"))
DATA_PATH = Path(os.getenv("DATA_PATH", FASE2_DIR / "data" / "diabetes_risk_prediction_dataset.csv"))
ACTIVE_VERSION = os.getenv("MODEL_VERSION", "v1")

# Esquema exacto de las 36 variables predictoras (permite None para aplicar imputación de Fase 2)
class PacienteInput(BaseModel):
    Age: Optional[float]
    Gender: Optional[str]
    Country: Optional[str]
    Height_cm: Optional[float]
    Weight_kg: Optional[float]
    BMI: Optional[float]
    Waist_Circumference_cm: Optional[float]
    Blood_Glucose: Optional[float]
    HbA1c: Optional[float]
    Fasting_Blood_Sugar: Optional[float]
    Insulin_Level: Optional[float]
    Blood_Pressure_Systolic: Optional[int]
    Blood_Pressure_Diastolic: Optional[int]
    Total_Cholesterol: Optional[float]
    HDL: Optional[float]
    LDL: Optional[float]
    Triglycerides: Optional[float]
    Heart_Rate: Optional[int]
    Physical_Activity_Level: Optional[str]
    Exercise_Hours_Per_Week: Optional[float]
    Daily_Walking_Minutes: Optional[float]
    Diet_Quality: Optional[str]
    Sugar_Intake_Level: Optional[str]
    Sleep_Hours: Optional[float]
    Stress_Level: Optional[str]
    Smoking_Status: Optional[str]
    Alcohol_Consumption: Optional[str]
    Family_History_Diabetes: Optional[str]
    Hypertension: Optional[str]
    Heart_Disease: Optional[str]
    Fatty_Liver: Optional[str]
    PCOS: Optional[str]
    Medication_Adherence: Optional[str]
    Work_Type: Optional[str]
    Residence_Type: Optional[str]
    Daily_Water_Intake_L: Optional[float]

@app.get("/health")
def health_check():
    return {"status": "ok", "mensaje": "API en ejecución", "version_activa": ACTIVE_VERSION}

@app.post("/predict")
def predict(paciente: PacienteInput):
    modelo_path = MODELS_DIR / ACTIVE_VERSION / "modelo.joblib"
    if not modelo_path.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"El modelo configurado ('{ACTIVE_VERSION}') no está disponible en {modelo_path}."
        )
    try:
        pipeline = joblib.load(modelo_path)
        df_fila = pd.DataFrame([paciente.model_dump()])
        prediccion = pipeline.predict(df_fila)[0]
        return {"prediccion": str(prediccion), "version": ACTIVE_VERSION}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno al predecir: {str(e)}")

@app.post("/train", status_code=201)
def train_new_version():
    if not DATA_PATH.is_file():
        raise HTTPException(status_code=503, detail=f"Dataset de entrenamiento no encontrado en {DATA_PATH}")
    
    # Buscar siguiente versión disponible sin sobrescribir (v1, v2, v3...)
    idx = 1
    while (MODELS_DIR / f"v{idx}").exists():
        idx += 1
    nueva_version = f"v{idx}"

    try:
        parser = build_parser()
        args = parser.parse_args([
            "--data", str(DATA_PATH),
            "--output-dir", str(MODELS_DIR),
            "--version", nueva_version,
            "--seed", "42"
        ])
        run_training(args)
        return {"mensaje": "Entrenamiento completado", "version": nueva_version}
    except TrainingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en entrenamiento: {str(e)}")
