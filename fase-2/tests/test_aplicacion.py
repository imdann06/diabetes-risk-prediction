
"""Pruebas automatizadas de la aplicación de riesgo de diabetes (Fase 2)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

# Permite importar los scripts ubicados en fase-2.
FASE2_DIR = Path(__file__).resolve().parents[1]
if str(FASE2_DIR) not in sys.path:
    sys.path.insert(0, str(FASE2_DIR))

import predict
import train


@pytest.fixture(scope="module")
def datos_entrenamiento(tmp_path_factory):
    """Crea datos pequeños para realizar las pruebas."""
    rng = np.random.default_rng(42)
    filas = []

    for clase, base in [("Low", 35), ("Moderate", 50), ("High", 65)]:
        for i in range(20):
            filas.append({
                "Age": float(base + rng.normal(0, 4)),
                "Glucose": float(base * 2 + rng.normal(0, 8)),
                "Activity": ["Low", "Medium", "High"][i % 3],
                "Diabetes_Risk": clase,
            })

    df = pd.DataFrame(filas)
    df.loc[2, "Age"] = np.nan
    df.loc[25, "Activity"] = np.nan

    work_dir = tmp_path_factory.mktemp("datos_prueba")
    csv_path = work_dir / "datos_sinteticos.csv"
    df.to_csv(csv_path, index=False)

    return csv_path, df


@pytest.fixture(scope="module")
def version_entrenada(datos_entrenamiento, tmp_path_factory):
    """Entrena un modelo pequeño para reutilizarlo en las pruebas."""
    csv_path, _ = datos_entrenamiento
    output_dir = tmp_path_factory.mktemp("modelos")

    args = train.build_parser().parse_args([
        "--data", str(csv_path),
        "--target", "Diabetes_Risk",
        "--output-dir", str(output_dir),
        "--version", "test-v1",
        "--seed", "42",
        "--test-size", "0.25",
        "--cv-folds", "2",
        "--n-estimators", "10",
        "--min-samples-leaf", "1",
        "--n-jobs", "1",
    ])

    version_dir = train.run_training(args)
    return args, version_dir


def test_entrenamiento_guarda_modelo_metricas_y_metadatos(version_entrenada):
    """Verifica que se guarden correctamente los archivos del entrenamiento."""
    args, version_dir = version_entrenada

    modelo_path = version_dir / "modelo.joblib"
    metricas_path = version_dir / "metrics.json"
    metadata_path = version_dir / "metadata.json"

    assert modelo_path.is_file()
    assert metricas_path.is_file()
    assert metadata_path.is_file()

    assert hasattr(joblib.load(modelo_path), "predict")

    metricas = json.loads(metricas_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert "f1_macro" in metricas["test"]["model"]
    assert "baseline_dummy_most_frequent" in metricas["test"]
    assert metadata["version"] == args.version
    assert metadata["target"] == "Diabetes_Risk"


def test_carga_y_prediccion_conservan_una_prediccion_por_fila(
    datos_entrenamiento, version_entrenada, tmp_path
):
    """Comprueba que el modelo genere una predicción por cada fila."""
    _, df = datos_entrenamiento
    _, version_dir = version_entrenada

    entrada = df.drop(columns=["Diabetes_Risk"]).iloc[:6].copy()
    entrada_path = tmp_path / "entrada.csv"
    salida_path = tmp_path / "predicciones.csv"

    entrada.to_csv(entrada_path, index=False)

    modelo = joblib.load(version_dir / "modelo.joblib")
    predicciones_esperadas = modelo.predict(entrada)

    predict.validar_y_predecir(
        str(version_dir / "modelo.joblib"),
        str(entrada_path),
        str(salida_path)
    )

    resultado = pd.read_csv(salida_path)

    assert len(resultado) == len(entrada)
    assert "prediccion" in resultado.columns
    assert resultado["prediccion"].tolist() == predicciones_esperadas.tolist()


def test_prediccion_admite_valores_nulos_en_columnas_con_imputacion(
    version_entrenada, tmp_path
):
    """Comprueba que el modelo pueda procesar valores nulos."""
    _, version_dir = version_entrenada

    entrada = pd.DataFrame({
        "Age": [np.nan, 42.0, 58.0],
        "Glucose": [110.0, np.nan, 180.0],
        "Activity": ["Low", "UnknownCategory", np.nan],
    })

    entrada_path = tmp_path / "entrada_con_nulos.csv"
    salida_path = tmp_path / "salida_con_nulos.csv"

    entrada.to_csv(entrada_path, index=False)

    predict.validar_y_predecir(
        str(version_dir / "modelo.joblib"),
        str(entrada_path),
        str(salida_path)
    )

    resultado = pd.read_csv(salida_path)

    assert len(resultado) == len(entrada)
    assert resultado["prediccion"].notna().all()


def test_entrada_invalida_sin_columna_requerida_muestra_error(
    version_entrenada, tmp_path, capsys
):
    """Comprueba que una entrada sin una columna obligatoria genere error."""
    _, version_dir = version_entrenada

    entrada = pd.DataFrame({
        "Age": [40.0],
        "Activity": ["Low"]
    })

    entrada_path = tmp_path / "entrada_invalida.csv"
    salida_path = tmp_path / "no_debe_crearse.csv"

    entrada.to_csv(entrada_path, index=False)

    with pytest.raises(SystemExit) as error:
        predict.validar_y_predecir(
            str(version_dir / "modelo.joblib"),
            str(entrada_path),
            str(salida_path)
        )

    assert error.value.code != 0

    mensaje = capsys.readouterr().out.lower()
    assert "error" in mensaje
    assert not salida_path.exists()


def test_versionamiento_no_sobrescribe_una_version_existente(version_entrenada):
    """Comprueba que una versión existente no sea sobrescrita."""
    args, version_dir = version_entrenada

    modelo_path = version_dir / "modelo.joblib"
    contenido_original = modelo_path.read_bytes()

    with pytest.raises(train.TrainingError, match="ya existe|No se sobrescribe"):
        train.run_training(args)

    assert modelo_path.read_bytes() == contenido_original
    assert (version_dir / "metrics.json").is_file()
    assert (version_dir / "metadata.json").is_file()
