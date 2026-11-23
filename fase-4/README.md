# Fase 4. Monitoreo de la API

Modelos y Simulación de Sistemas I · Universidad de Antioquia · Grupo G27

Esta fase agrega a la API de `fase-3/` un registro de cada llamada a `POST /predict`
(incluidas las rechazadas por validación) y un script que calcula dos métricas de operación.

## 1. Arranque con el registro habilitado

El registro lo escribe `fase-4/registro.py`, que `fase-3/api.py` invoca desde un middleware
en cada llamada a `POST /predict` (válida o rechazada: 200, 400, 422, 503, 500). Se usa el
`Dockerfile` de `fase-3/`, que copia todo el repositorio (incluida `fase-4/`); por eso se
construye desde la **raíz** del repo.

```bash
# 1) Imagen
docker build -f fase-3/Dockerfile -t diabetes-api .

# 2) Arrancar con la versión v1 y el registro en una carpeta montada
docker run --rm -p 8000:8000 \
  -e MODEL_VERSION=v1 \
  -v "$(pwd)/fase-2/models:/app/fase-2/models" \
  -v "$(pwd)/fase-4/logs:/app/fase-4/logs" \
  diabetes-api
```

En PowerShell reemplace `$(pwd)` por `${PWD}` y escriba el comando en una sola línea.

- **Versión activa:** variable de entorno `MODEL_VERSION` (por defecto `v1`).
- **Registro persistente:** `fase-4/logs/logs_predicciones.csv` en el equipo (dentro del
  contenedor es `/app/fase-4/logs/`). Columnas: `fecha_hora,estado_http,version,prediccion`.
  Se abre en modo agregar, así que se conserva entre ejecuciones; el encabezado se escribe
  al crear el archivo y cada intento genera una sola fila. La ruta puede cambiarse con
  `LOG_PREDICCIONES`.

## 2. Ensayo y script de monitoreo

Con la API en ejecución, desde la raíz del repo:

```bash
python fase-4/ensayo_fase4.py                                    # 2 válidas + 1 sin "Age"
python fase-4/monitoreo.py --log fase-4/logs/logs_predicciones.csv
```

`ensayo_fase4.py` envía las dos filas de `fase-2/data/ejemplo_prediccion.csv` y una tercera sin
`Age`. `monitoreo.py` muestra el total de solicitudes y el porcentaje de fallidas (HTTP
400–599). Con el registro vacío muestra total 0 y «sin solicitudes».

> Use un registro vacío (borre `fase-4/logs/logs_predicciones.csv`) antes del ensayo, para
> que el reporte corresponda solo a esas 3 solicitudes.

## 3. Versiones y cómo volver a la anterior

| Identificador | Ubicación | Estado |
|---|---|---|
| `v1` | `fase-2/models/v1/` | **Activa** (semilla 42, F1-macro 0,6488) |
| `v2` | `fase-2/models/v2/` | Disponible (semilla 7, F1-macro 0,6497) |

Cambiar de versión no requiere modificar código; solo la configuración existente:

```bash
# Docker: reiniciar el contenedor con otra versión
docker run --rm -p 8000:8000 -e MODEL_VERSION=v2 \
  -v "$(pwd)/fase-2/models:/app/fase-2/models" \
  -v "$(pwd)/fase-4/logs:/app/fase-4/logs" diabetes-api

# Local (desde fase-3/)
MODEL_VERSION=v2 uvicorn api:app --host 0.0.0.0 --port 8000      # bash
$env:MODEL_VERSION="v2"; uvicorn api:app --host 0.0.0.0 --port 8000   # PowerShell
```

Para volver a la anterior, repetir con `MODEL_VERSION=v1`. `GET /health` informa la versión
activa. Si la versión pedida no existe en `fase-2/models/`, `/predict` responde 503. Las
versiones creadas con `POST /train` (v3, v4, …) se activan igual, con `MODEL_VERSION=vN`.

## 4. Política de reentrenamiento

Cuando el equipo reúna 1.000 casos nuevos con el nivel de riesgo real confirmado, o cuando
cambie el origen de los datos, quien ejecuta `POST /train` (el responsable de modelado)
entrenará una versión candidata sin tocar v1 ni v2. Se compararán la candidata y la versión
activa con F1-macro (la métrica de fase 2) sobre los mismos casos reservados, mirando también
el recall de `Low`; la candidata se activa con `MODEL_VERSION` solo si mejora F1-macro y no
empeora ese recall. Mientras no existan resultados reales, la tasa de fallos solo describe el
servicio y no la calidad predictiva.

## 5. Reporte

Ensayo ejecutado el 22/11/2026 con la API en Docker (`diabetes-api`), versión activa `v1` y el
registro vacío al empezar. Las horas del registro están en UTC (hora del contenedor), por eso
aparecen como 23/11 a las 03:57 (22:57 en Colombia).

**Solicitudes enviadas** (`python fase-4/ensayo_fase4.py`)

| # | Entrada | Respuesta |
|---|---|---|
| 1 | Fila 1 de `ejemplo_prediccion.csv` convertida a JSON | HTTP 200, predicción `High`, versión `v1` |
| 2 | Fila 2 de `ejemplo_prediccion.csv` convertida a JSON | HTTP 200, predicción `High`, versión `v1` |
| 3 | Fila 1 sin el campo `Age` | HTTP 422, `{"detail":[{"type":"missing","loc":["body","Age"],"msg":"Field required", ...}]}` |

**Registro obtenido** (`fase-4/logs/logs_predicciones.csv`; copia en `fase-4/muestra_logs_predicciones.csv`)

```text
fecha_hora,estado_http,version,prediccion
2026-11-23T03:57:57,200,v1,High
2026-11-23T03:57:58,200,v1,High
2026-11-23T03:57:58,422,,
```

**Salida del script de monitoreo** (`python fase-4/monitoreo.py --log fase-4/logs/logs_predicciones.csv`)

```text
Registro: fase-4\logs\logs_predicciones.csv
Total de solicitudes: 3
Solicitudes fallidas (HTTP 4xx/5xx): 1
Porcentaje de fallos: 33,3 %
```

**Indicadores**

| Indicador | Valor |
|---|---|
| Total de solicitudes | 3 |
| Solicitudes fallidas (4xx/5xx) | 1 |
| Porcentaje de fallos | 33,3 % |

**Interpretación.** La API procesó dos registros con `v1` y rechazó el tercero porque faltaba el
campo `Age`. La tasa de fallos fue 1/3 (33,3 %); el rechazo era el resultado esperado de la
prueba de validación y no indica un problema del modelo. Cada intento generó una sola fila,
incluido el rechazado, que quedó con versión y predicción vacías.

**Versiones identificadas:** `v1` (activa) y `v2` (disponible), ambas en `fase-2/models/`.
