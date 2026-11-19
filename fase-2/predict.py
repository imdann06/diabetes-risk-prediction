import argparse
import sys
import os
import pandas as pd
import joblib

COLUMNAS_NUMERICAS = [
    "Age", "Height_cm", "Weight_kg", "BMI", "Waist_Circumference_cm",
    "Blood_Glucose", "HbA1c", "Fasting_Blood_Sugar", "Insulin_Level",
    "Blood_Pressure_Systolic", "Blood_Pressure_Diastolic", "Total_Cholesterol",
    "HDL", "LDL", "Triglycerides", "Heart_Rate", "Exercise_Hours_Per_Week",
    "Daily_Walking_Minutes", "Sleep_Hours", "Daily_Water_Intake_L"
]

COLUMNAS_CATEGORICAS = [
    "Gender", "Country", "Physical_Activity_Level", "Diet_Quality",
    "Sugar_Intake_Level", "Stress_Level", "Smoking_Status", "Alcohol_Consumption",
    "Family_History_Diabetes", "Hypertension", "Heart_Disease", "Fatty_Liver",
    "PCOS", "Medication_Adherence", "Work_Type", "Residence_Type"
]

COLUMNAS_ESPERADAS = COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS


def cargar_modelo(ruta_modelo):
    if not os.path.exists(ruta_modelo):
        print(f"Error: El archivo del modelo no existe en la ruta: {ruta_modelo}")
        sys.exit(1)
    try:
        pipeline = joblib.load(ruta_modelo)
        return pipeline
    except Exception as e:
        print(f"Error al cargar el artefacto del modelo: {e}")
        sys.exit(1)


def validar_esquema_y_tipos(df_in):
    # 1. Validar que no falte ninguna columna obligatoria
    faltantes = [col for col in COLUMNAS_ESPERADAS if col not in df_in.columns]
    if faltantes:
        print(f"Error de esquema: Faltan columnas requeridas en el CSV: {faltantes}")
        sys.exit(1)

    # 2. Validar tipos numéricos (permitiendo nulos/NaN para que el pipeline los impute)
    for col in COLUMNAS_NUMERICAS:
        serie_convertida = pd.to_numeric(df_in[col], errors="coerce")
        # Si aparecen nuevos NaN tras convertir a número, significa que había texto inválido
        if serie_convertida.isna().sum() > df_in[col].isna().sum():
            print(f"Error de tipo: La columna numérica '{col}' contiene valores no numéricos.")
            sys.exit(1)

    # 3. Validar que las columnas categóricas no vengan solo con números puros
    for col in COLUMNAS_CATEGORICAS:
        if pd.api.types.is_numeric_dtype(df_in[col].dropna()):
            print(f"Error de tipo: La columna categórica '{col}' debe contener texto, no valores numéricos.")
            sys.exit(1)


def validar_y_predecir(ruta_modelo, ruta_entrada, ruta_salida):
    pipeline = cargar_modelo(ruta_modelo)
    
    if not os.path.exists(ruta_entrada):
        print(f"Error: El archivo de entrada no existe: {ruta_entrada}")
        sys.exit(1)
        
    try:
        df_in = pd.read_csv(ruta_entrada)
    except Exception as e:
        print(f"Error al leer el CSV de entrada: {e}")
        sys.exit(1)
        
    if df_in.empty:
        print("Error: El archivo CSV de entrada está vacío.")
        sys.exit(1)

    validar_esquema_y_tipos(df_in)

    try:
        predicciones = pipeline.predict(df_in[COLUMNAS_ESPERADAS])
    except Exception as e:
        print(f"Error durante la inferencia/predicción: {e}")
        sys.exit(1)

    df_out = df_in.copy()
    df_out['prediccion'] = predicciones
    
    directorio_salida = os.path.dirname(ruta_salida)
    if directorio_salida and not os.path.exists(directorio_salida):
        os.makedirs(directorio_salida, exist_ok=True)
        
    df_out.to_csv(ruta_salida, index=False)
    print(f"Predicciones generadas exitosamente ({len(df_out)} filas). Guardadas en: {ruta_salida}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de inferencia/predicción - Proyecto Diabetes Risk")
    parser.add_argument("--model", type=str, required=True, help="Ruta al archivo .joblib del modelo o pipeline")
    parser.add_argument("--input", type=str, required=True, help="Ruta al archivo CSV con datos de entrada")
    parser.add_argument("--output", type=str, required=True, help="Ruta donde se guardará el CSV con predicciones")
    
    args = parser.parse_args()
    validar_y_predecir(args.model, args.input, args.output)
