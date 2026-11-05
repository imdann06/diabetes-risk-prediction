import argparse
import sys
import os
import pandas as pd
import joblib

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

    try:
        predicciones = pipeline.predict(df_in)
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
