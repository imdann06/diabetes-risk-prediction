"""Registro de las solicitudes a POST /predict (Fase 4).

Cada llamada a /predict, válida o rechazada, genera UNA fila en un CSV:

    fecha_hora,estado_http,version,prediccion
    2026-11-15T10:00:00,200,v1,High
    2026-11-15T10:01:00,422,,

- En un error (400/422/503/500) 'prediccion' queda vacía y 'version' también si
  no se llegó a usar un modelo.
- El archivo se abre en modo agregar: el historial se conserva entre ejecuciones.
- El encabezado se escribe solo cuando el archivo se crea (o está vacío).
- Ruta por defecto: logs/logs_predicciones.csv, cambiable con la variable de
  entorno LOG_PREDICCIONES. En Docker conviene montar la carpeta logs/ para que
  el registro persista fuera del contenedor.
"""
from __future__ import annotations

import csv
import os
import sys
import threading
from datetime import datetime
from pathlib import Path

COLUMNAS = ["fecha_hora", "estado_http", "version", "prediccion"]
RUTA_POR_DEFECTO = "logs/logs_predicciones.csv"
_CERROJO = threading.Lock()  # evita filas mezcladas si llegan solicitudes simultáneas


def ruta_log(ruta: str | Path | None = None) -> Path:
    """Ruta del registro: argumento, variable LOG_PREDICCIONES o valor por defecto."""
    return Path(ruta or os.environ.get("LOG_PREDICCIONES", RUTA_POR_DEFECTO))


def registrar_prediccion(estado_http: int, version: str | None = None,
                         prediccion=None, ruta: str | Path | None = None) -> None:
    """Agrega una fila al registro. Si no se puede escribir, avisa sin tumbar la API."""
    destino = ruta_log(ruta)
    fila = [
        datetime.now().isoformat(timespec="seconds"),
        int(estado_http),
        "" if version is None else str(version),
        "" if prediccion is None else str(prediccion),
    ]
    try:
        with _CERROJO:
            destino.parent.mkdir(parents=True, exist_ok=True)
            nuevo = not destino.exists() or destino.stat().st_size == 0
            with open(destino, "a", newline="", encoding="utf-8") as f:
                escritor = csv.writer(f)
                if nuevo:
                    escritor.writerow(COLUMNAS)
                escritor.writerow(fila)
    except OSError as exc:
        print(f"ADVERTENCIA: no se pudo escribir el registro en {destino}: {exc}",
              file=sys.stderr)
