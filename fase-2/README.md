# Fase 2. Entrenamiento y aplicación del modelo

Modelos y Simulación de Sistemas I · Universidad de Antioquia · Grupo G27

Clasificación multiclase del riesgo de diabetes (`Low`, `Moderate`, `High`) con un
`RandomForestClassifier`. El preprocesamiento (imputación, escalamiento y one-hot) va
dentro del mismo `Pipeline` guardado, por lo que `predict.py` aplica exactamente las
transformaciones aprendidas en el entrenamiento.

> Los datos son **sintéticos** (Kaggle, 50.000 registros). Los resultados son académicos y
> no constituyen diagnóstico ni validación clínica.

## 1. Estructura

```text
fase-2/
├── data/ejemplo_prediccion.csv      # 2 registros sin la variable objetivo
├── models/v1/, models/v2/           # modelo.joblib + metrics.json + metadata.json
├── tests/test_aplicacion.py         # 5 pruebas con pytest
├── train.py   predict.py
├── requirements.txt   Dockerfile   .dockerignore
```

Todos los comandos de este README se ejecutan **desde la raíz del repositorio**
(`diabetes-risk-prediction/`), salvo que se indique otra cosa.

## 2. Preparar el entorno y los datos

```bash
python -m pip install -r fase-2/requirements.txt
```

El CSV completo de entrenamiento está comprimido dos veces en
`fase-1/data/diabetes_risk_prediction_dataset.zip.zip`. Para entrenar, extráigalo en
`fase-2/data/` (no es necesario para predecir):

```bash
unzip fase-1/data/diabetes_risk_prediction_dataset.zip.zip -d fase-2/data/
# Resultado: fase-2/data/diabetes_risk_prediction_dataset.csv
```

## 3. Entrenar (`train.py`)

```bash
cd fase-2
python train.py --data data/diabetes_risk_prediction_dataset.csv --target Diabetes_Risk \
    --output-dir models --version v3 --seed 42
```

- `--version vN` crea `models/vN/`. Si ya existe, falla con `TrainingError` y **no sobrescribe**.
- `--version auto` asigna el siguiente identificador libre (v3, v4, …).
- Opciones adicionales: `python train.py --help`.
- Antes de terminar, el script compara las predicciones del modelo en memoria con las del
  archivo recién guardado y vuelve a cargarlo (comprobación guardado/carga).

## 4. Predecir (`predict.py`)

```bash
cd fase-2
python predict.py --model models/v1/modelo.joblib \
    --input data/ejemplo_prediccion.csv --output salida/predicciones.csv
```

Genera `salida/predicciones.csv` con las columnas de entrada más `prediccion`. No requiere
la variable objetivo.

### Contrato de entrada

El CSV debe contener **las 36 columnas** siguientes (el orden no importa):

| Tipo | Columnas |
|---|---|
| Numéricas (20) | `Age`, `Height_cm`, `Weight_kg`, `BMI`, `Waist_Circumference_cm`, `Blood_Glucose`, `HbA1c`, `Fasting_Blood_Sugar`, `Insulin_Level`, `Blood_Pressure_Systolic`, `Blood_Pressure_Diastolic`, `Total_Cholesterol`, `HDL`, `LDL`, `Triglycerides`, `Heart_Rate`, `Exercise_Hours_Per_Week`, `Daily_Walking_Minutes`, `Sleep_Hours`, `Daily_Water_Intake_L` |
| Categóricas, texto (16) | `Gender`, `Country`, `Physical_Activity_Level`, `Diet_Quality`, `Sugar_Intake_Level`, `Stress_Level`, `Smoking_Status`, `Alcohol_Consumption`, `Family_History_Diabetes`, `Hypertension`, `Heart_Disease`, `Fatty_Liver`, `PCOS`, `Medication_Adherence`, `Work_Type`, `Residence_Type` |

| Situación | Comportamiento |
|---|---|
| Falta una columna | Mensaje `Error de esquema: Faltan columnas requeridas …` con los nombres; código de salida 1; no se crea la salida. |
| Texto en una columna numérica (p. ej. `Age = "abc"`) | `Error de tipo: La columna numérica 'Age' contiene valores no numéricos.`; código 1. |
| Número en una columna categórica | `Error de tipo: La columna categórica '…' debe contener texto …`; código 1. |
| Valor nulo (celda vacía) | Permitido. Se imputa con la regla guardada: **mediana** (numéricas) o **moda** (categóricas). |
| Categoría no vista en el entrenamiento | Se ignora (`OneHotEncoder(handle_unknown="ignore")`). |

Columnas excluidas del modelo: `Patient_ID`, `Diabetes_Risk_Score`,
`Doctor_Consultation_Needed`, `AI_Health_Recommendation` (identificador o derivadas del riesgo).

## 5. Pruebas

```bash
cd fase-2
python -m pytest -v
```

Cinco pruebas: guardado de modelo/métricas/metadatos, una predicción por fila, valores
nulos, entrada sin columna obligatoria y protección contra sobrescritura de versiones.

**Estado actual (22/11/2026, ejecutado en Docker): 3 pasan y 2 fallan.**

| Prueba | Resultado |
|---|---|
| `test_entrenamiento_guarda_modelo_metricas_y_metadatos` | PASSED |
| `test_carga_y_prediccion_conservan_una_prediccion_por_fila` | **FAILED** |
| `test_prediccion_admite_valores_nulos_en_columnas_con_imputacion` | **FAILED** |
| `test_entrada_invalida_sin_columna_requerida_muestra_error` | PASSED |
| `test_versionamiento_no_sobrescribe_una_version_existente` | PASSED |

**Problema conocido.** Desde que `predict.py` valida el esquema estricto (exige las 36
columnas de `COLUMNAS_ESPERADAS`), las dos pruebas que fallan terminan con
`SystemExit: 1` y el mensaje `Error de esquema: Faltan columnas requeridas en el CSV`.
Ambas entrenan un modelo de juguete con solo 3 columnas (`Age`, `Glucose`, `Activity`), que la
validación rechaza. El fallo no depende de Docker: el modelo real (v1) sí predice (sección 6).
Corrección pendiente: que el modelo de prueba use las 36 columnas del proyecto.

## 6. Docker

El `Dockerfile` está en `fase-2/` y su contexto de construcción es `fase-2/`. El `CMD` por
defecto muestra la ayuda de `predict.py`.

```bash
# 1) Construir la imagen
docker build -t diabetes-fase2 ./fase-2

# 2) Comprobar que el contenedor arranca
docker run --rm diabetes-fase2

# 3) Predecir (modelos y datos montados desde el equipo; la salida queda en fase-2/salida/)
docker run --rm \
  -v "$(pwd)/fase-2/models:/app/models" \
  -v "$(pwd)/fase-2/data:/app/data" \
  -v "$(pwd)/fase-2/salida:/app/salida" \
  diabetes-fase2 \
  python predict.py --model models/v1/modelo.joblib \
    --input data/ejemplo_prediccion.csv --output salida/predicciones.csv

# 4) Entrenar una versión nueva (requiere el CSV completo en fase-2/data/, ver sección 2)
docker run --rm \
  -v "$(pwd)/fase-2/models:/app/models" \
  -v "$(pwd)/fase-2/data:/app/data" \
  diabetes-fase2 \
  python train.py --data data/diabetes_risk_prediction_dataset.csv --target Diabetes_Risk \
    --output-dir models --version v3 --seed 42

# 5) Pruebas dentro del contenedor
docker run --rm diabetes-fase2 python -m pytest -v
```

En PowerShell reemplace `$(pwd)` por `${PWD}`. Persistencia: como `models/` está montado,
`fase-2/models/v3/` aparece en el equipo al terminar el contenedor.

### Evidencia de ejecución en el contenedor

Ejecutado el 22/11/2026 en Windows con Docker Desktop (Docker 29.9.3). La imagen usa
Python 3.11.18, mientras que v1 y v2 se entrenaron con Python 3.13.18 y scikit-learn 1.6.1:
el modelo cargó y predijo sin errores.

**Construcción de la imagen (paso 1):** completada con `docker build -t diabetes-fase2 ./fase-2`.


**Inicio del contenedor (paso 2)**

```text
usage: predict.py [-h] --model MODEL --input INPUT --output OUTPUT

Script de inferencia/predicción - Proyecto Diabetes Risk

options:
  -h, --help       show this help message and exit
  --model MODEL    Ruta al archivo .joblib del modelo o pipeline
  --input INPUT    Ruta al archivo CSV con datos de entrada
  --output OUTPUT  Ruta donde se guardará el CSV con predicciones
```

**Predicción con v1 (paso 3)**

```text
Predicciones generadas exitosamente (2 filas). Guardadas en: salida/predicciones.csv
```

Contenido de `fase-2/salida/predicciones.csv` (columna `prediccion`):



**Pruebas dentro del contenedor (paso 5)** — salida resumida (se omite la lista de columnas del mensaje de error):

```text
platform linux -- Python 3.11.18, pytest-9.2.0
collected 5 items

tests/test_aplicacion.py::test_entrenamiento_guarda_modelo_metricas_y_metadatos PASSED
tests/test_aplicacion.py::test_carga_y_prediccion_conservan_una_prediccion_por_fila FAILED
tests/test_aplicacion.py::test_prediccion_admite_valores_nulos_en_columnas_con_imputacion FAILED
tests/test_aplicacion.py::test_entrada_invalida_sin_columna_requerida_muestra_error PASSED
tests/test_aplicacion.py::test_versionamiento_no_sobrescribe_una_version_existente PASSED

E   SystemExit: 1
Error de esquema: Faltan columnas requeridas en el CSV: ['Height_cm', 'Weight_kg', ... ]
/app/predict.py:42: SystemExit: 1

2 failed, 3 passed in 4.51s
```

Ver «Problema conocido» en la sección 5.

**Entrenamiento y persistencia (paso 4):** pendiente de ejecutar en el contenedor. Requiere extraer
el CSV completo en `fase-2/data/` (sección 2).

## 7. Trazabilidad de versiones

| | v1 | v2 |
|---|---|---|
| Semilla (`random_state`, también de la partición) | 42 | 7 |
| Algoritmo / parámetros | RandomForest, 100 árboles, `class_weight=balanced`, `min_samples_leaf=5` | igual |
| Datos (SHA-256 del CSV) | `131438e6…565b3f05` (50.000 filas, 36 variables) | igual |
| Partición | 40.000 entrenamiento / 10.000 prueba, estratificada | igual, **otra semilla** |
| F1-macro (prueba) | 0,6488 | 0,6497 |
| Accuracy (prueba) | 0,9098 | 0,9049 |
| Recall de `Low` (94 casos) | 0,096 | 0,106 |
| Entorno | Python 3.13.18, scikit-learn 1.6.1 | igual |
| Commit del código | no registrado en `metadata.json` (`git_commit: null`) | no registrado |

**Cómo interpretar la diferencia.** v1 y v2 usan semillas distintas, y la semilla también
cambia la partición entrenamiento/prueba; por eso evalúan sobre conjuntos de prueba
diferentes. La diferencia de 0,001 en F1-macro está dentro de la variación esperada entre
semillas y **no** se interpreta como mejora del modelo. Además, `Low` tiene solo 94 casos
de prueba, por lo que su recall es muy inestable.

> Nota: los `metadata.json` actuales no incluyen el commit. Las versiones nuevas deben
> entrenarse con `git` disponible para que `git_commit` quede registrado, y anotarse aquí.
