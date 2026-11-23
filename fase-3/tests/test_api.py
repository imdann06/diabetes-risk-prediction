"""Pruebas automatizadas de la API (Fase 3).

Se ejecutan desde la carpeta fase-3 con:  python -m pytest -v

Para no depender del modelo grande (v1, ~13 MB) se entrena un modelo pequeño en una
carpeta temporal con las mismas 36 columnas del proyecto, y se le indica a la API que
lo use. Así las pruebas son rápidas y no tocan fase-2/models.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

FASE3_DIR = Path(__file__).resolve().parents[1]
FASE2_DIR = FASE3_DIR.parent / "fase-2"
for ruta in (FASE3_DIR, FASE2_DIR):
    if str(ruta) not in sys.path:
        sys.path.insert(0, str(ruta))

import api  # noqa: E402
import train  # noqa: E402

EJEMPLO_CSV = FASE2_DIR / "data" / "ejemplo_prediccion.csv"
VERSION_PRUEBA = "test-v1"


def _fila_ejemplo(indice: int = 1) -> dict:
    """Convierte una fila real de ejemplo_prediccion.csv en un JSON (NaN -> null)."""
    df = pd.read_csv(EJEMPLO_CSV)
    return json.loads(df.iloc[[indice]].to_json(orient="records"))[0]


def _dataset_pequeno(ruta_csv: Path) -> None:
    """Dataset sintético mínimo con las mismas columnas que ejemplo_prediccion.csv."""
    rng = np.random.default_rng(42)
    base = pd.read_csv(EJEMPLO_CSV)
    columnas_num = base.select_dtypes(include="number").columns
    columnas_cat = [c for c in base.columns if c not in columnas_num]

    filas = []
    for clase, factor in [("Low", 0.8), ("Moderate", 1.0), ("High", 1.3)]:
        for _ in range(20):
            fila = {}
            for c in columnas_num:
                valor = base[c].mean() * factor
                fila[c] = float(valor * (1 + rng.normal(0, 0.05)))
            for c in columnas_cat:
                fila[c] = str(rng.choice(base[c].unique()))
            fila["Diabetes_Risk"] = clase
            filas.append(fila)
    pd.DataFrame(filas).to_csv(ruta_csv, index=False)


@pytest.fixture(scope="module")
def cliente(tmp_path_factory):
    """Cliente de pruebas de FastAPI apuntando a un modelo pequeño temporal."""
    trabajo = tmp_path_factory.mktemp("api_modelos")
    csv_path = trabajo / "datos_pequenos.csv"
    _dataset_pequeno(csv_path)

    args = train.build_parser().parse_args([
        "--data", str(csv_path),
        "--target", "Diabetes_Risk",
        "--output-dir", str(trabajo / "models"),
        "--version", VERSION_PRUEBA,
        "--seed", "42",
        "--test-size", "0.25",
        "--cv-folds", "2",
        "--n-estimators", "10",
        "--min-samples-leaf", "1",
        "--n-jobs", "1",
    ])
    train.run_training(args)

    mp = pytest.MonkeyPatch()
    mp.setattr(api, "MODELS_DIR", trabajo / "models")
    mp.setattr(api, "ACTIVE_VERSION", VERSION_PRUEBA)
    # Fase 4: /predict registra cada llamada; se redirige el registro a una carpeta
    # temporal para que las pruebas no ensucien fase-4/logs/logs_predicciones.csv.
    mp.setenv("LOG_PREDICCIONES", str(trabajo / "logs" / "logs_predicciones.csv"))
    try:
        yield TestClient(api.app)
    finally:
        mp.undo()


def test_health_responde_200(cliente):
    """GET /health devuelve 200 y un JSON que indica que la API está en ejecución."""
    respuesta = cliente.get("/health")

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["status"] == "ok"


def test_predict_valido_devuelve_prediccion_y_version(cliente):
    """Un registro válido (fila real del CSV de ejemplo) devuelve 200, predicción y versión."""
    respuesta = cliente.post("/predict", json=_fila_ejemplo())

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert "prediccion" in cuerpo and "version" in cuerpo
    assert cuerpo["prediccion"] in {"Low", "Moderate", "High"}
    assert cuerpo["version"] == VERSION_PRUEBA


def test_predict_sin_campo_obligatorio_se_rechaza(cliente):
    """Si falta una variable obligatoria se responde 422 y el mensaje nombra el campo."""
    registro = _fila_ejemplo()
    del registro["Age"]

    respuesta = cliente.post("/predict", json=registro)

    assert respuesta.status_code == 422
    assert "Age" in respuesta.text
