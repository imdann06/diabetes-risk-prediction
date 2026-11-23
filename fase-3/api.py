import os
import sys
from pathlib import Path
from typing import Optional
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

# Permitir importar módulos de fase-2
BASE_DIR = Path(__file__).resolve().parent.parent
FASE2_DIR = BASE_DIR / "fase-2"
sys.path.append(str(FASE2_DIR))

# Fase 4: el registro de /predict vive en fase-4 y se guarda en fase-4/logs (carpeta montable en Docker)
FASE4_DIR = BASE_DIR / "fase-4"
sys.path.append(str(FASE4_DIR))
os.environ.setdefault("LOG_PREDICCIONES", str(FASE4_DIR / "logs" / "logs_predicciones.csv"))

from train import build_parser, run_training, TrainingError
from registro import registrar_prediccion

app = FastAPI(title="API Riesgo de Diabetes - Fase 3", version="1.0.0")


@app.middleware("http")
async def registrar_predict(request: Request, call_next):
    """Guarda una fila en el registro por cada llamada a POST /predict (también 400, 422, 503 y 500)."""
    if request.method != "POST" or request.url.path != "/predict":
        return await call_next(request)
    try:
        response = await call_next(request)
    except Exception:
        registrar_prediccion(500)
        raise
    registrar_prediccion(
        response.status_code,
        getattr(request.state, "version", None),
        getattr(request.state, "prediccion", None),
    )
    return response


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
def predict(request: Request, paciente: PacienteInput):
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
        # request.state deja version y prediccion disponibles para el registro (middleware)
        request.state.version = ACTIVE_VERSION
        request.state.prediccion = str(prediccion)
        return {"prediccion": str(prediccion), "version": ACTIVE_VERSION}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno al predecir: {str(e)}")

@app.post("/train", status_code=201)
def train_new_version():
    if not DATA_PATH.is_file():
        raise HTTPException(status_code=503, detail=f"Dataset de entrenamiento no encontrado en {DATA_PATH}")

    try:
        parser = build_parser()
        args = parser.parse_args([
            "--data", str(DATA_PATH),
            "--output-dir", str(MODELS_DIR),
            "--version", "auto",  # train.py asigna el siguiente identificador libre (v3, v4...)
            "--seed", "42"
        ])
        version_dir = run_training(args)
        return {"mensaje": "Entrenamiento completado", "version": version_dir.name}
    except TrainingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en entrenamiento: {str(e)}")

