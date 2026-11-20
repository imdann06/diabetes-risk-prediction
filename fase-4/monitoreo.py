"""Métricas de operación de la API a partir del registro de /predict (Fase 4).

Uso (desde la carpeta fase-4, en una sola línea):

    python monitoreo.py --log logs/logs_predicciones.csv

Muestra:
  - el total de solicitudes registradas;
  - el porcentaje de solicitudes fallidas (respuestas HTTP 4xx y 5xx).

Si el registro no tiene solicitudes, muestra total 0 y «sin solicitudes».
Si no se indica --log se usa LOG_PREDICCIONES o logs/logs_predicciones.csv.
Termina con código distinto de cero si el archivo no existe o no se puede leer.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

RUTA_POR_DEFECTO = "logs/logs_predicciones.csv"


class MonitoreoError(Exception):
    """Error previsible al leer el registro."""


def leer_estados(ruta: Path) -> tuple[list[int], int]:
    """Devuelve (códigos HTTP de cada fila, cantidad de filas ilegibles)."""
    if not ruta.is_file():
        raise MonitoreoError(f"No existe el registro: {ruta}")
    estados: list[int] = []
    ilegibles = 0
    try:
        with open(ruta, newline="", encoding="utf-8") as f:
            lector = csv.DictReader(f)
            if lector.fieldnames is None:  # archivo vacío: sin solicitudes
                return estados, ilegibles
            if "estado_http" not in lector.fieldnames:
                raise MonitoreoError(
                    f"El registro {ruta} no tiene la columna 'estado_http'. "
                    f"Columnas encontradas: {lector.fieldnames}")
            for fila in lector:
                try:
                    estados.append(int(str(fila["estado_http"]).strip()))
                except (TypeError, ValueError):
                    ilegibles += 1
    except OSError as exc:
        raise MonitoreoError(f"No se pudo leer el registro {ruta}: {exc}") from exc
    return estados, ilegibles


def calcular_metricas(estados: list[int]) -> dict:
    """Total de solicitudes, cantidad fallida (400-599) y porcentaje de fallos."""
    total = len(estados)
    fallidas = sum(1 for e in estados if 400 <= e <= 599)
    porcentaje = (fallidas / total * 100) if total else None
    return {"total": total, "fallidas": fallidas, "porcentaje_fallos": porcentaje}


def formatear_porcentaje(porcentaje: float | None) -> str:
    if porcentaje is None:
        return "sin solicitudes"
    return f"{porcentaje:.1f} %".replace(".", ",")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calcula el total de solicitudes y el porcentaje de fallos de /predict.")
    parser.add_argument("--log", default=os.environ.get("LOG_PREDICCIONES", RUTA_POR_DEFECTO),
                        help=f"Ruta del registro CSV (por defecto: {RUTA_POR_DEFECTO}).")
    args = parser.parse_args(argv)
    try:
        estados, ilegibles = leer_estados(Path(args.log))
    except MonitoreoError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    m = calcular_metricas(estados)
    print(f"Registro: {args.log}")
    print(f"Total de solicitudes: {m['total']}")
    print(f"Solicitudes fallidas (HTTP 4xx/5xx): {m['fallidas']}")
    print(f"Porcentaje de fallos: {formatear_porcentaje(m['porcentaje_fallos'])}")
    if ilegibles:
        print(f"Aviso: {ilegibles} fila(s) sin código HTTP válido no se contaron.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
