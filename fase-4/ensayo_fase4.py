"""Ensayo de la Fase 4: 2 solicitudes válidas + 1 inválida contra la API en ejecución.

Uso (con la API ya levantada en http://localhost:8000, desde la raíz del repo):

    python fase-4/ensayo_fase4.py

Envía las 2 filas de fase-2/data/ejemplo_prediccion.csv convertidas a JSON y una
tercera fila sin el campo obligatorio "Age". Resultado esperado: 200, 200 y 422.
Después ejecute:  python fase-4/monitoreo.py --log fase-4/logs/logs_predicciones.csv
"""
import csv
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

URL = "http://localhost:8000/predict"
CSV_EJEMPLO = Path(__file__).resolve().parents[1] / "fase-2" / "data" / "ejemplo_prediccion.csv"


def convertir(valor: str):
    """Celda del CSV -> valor JSON: vacío = null, número = número, resto = texto."""
    if valor == "":
        return None
    try:
        return int(valor)
    except ValueError:
        pass
    try:
        return float(valor)
    except ValueError:
        return valor


def enviar(registro: dict) -> tuple[int, str]:
    req = urllib.request.Request(
        URL, data=json.dumps(registro).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:  # 4xx/5xx también son una respuesta válida del ensayo
        return exc.code, exc.read().decode("utf-8")
    except urllib.error.URLError as exc:
        sys.exit(f"No se pudo conectar con {URL}: {exc}. ¿Está la API en ejecución?")


def main() -> None:
    with open(CSV_EJEMPLO, newline="", encoding="utf-8") as f:
        filas = [{k: convertir(v) for k, v in fila.items()} for fila in csv.DictReader(f)]

    sin_edad = dict(filas[0])
    del sin_edad["Age"]
    solicitudes = [
        ("1. válida (fila 1 del CSV)", filas[0]),
        ("2. válida (fila 2 del CSV)", filas[1]),
        ("3. inválida (sin Age)", sin_edad),
    ]
    for nombre, registro in solicitudes:
        codigo, cuerpo = enviar(registro)
        print(f"{nombre}: HTTP {codigo} -> {cuerpo[:200]}")


if __name__ == "__main__":
    main()
