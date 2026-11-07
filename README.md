# diabetes-risk-prediction
Proyecto de Machine Learning para predecir el nivel de riesgo de diabetes a partir de factores clínicos, demográficos y de estilo de vida.

## Integrantes

- Daniel Diaz Escobar
- Maria Fernanda Echeverri Osorio
- Salome Toro Cadavid

## Fuente y naturaleza del conjunto de datos

Se utiliza el conjunto **Diabetes Risk Prediction Dataset (50K patients)**, disponible en Kaggle:

https://www.kaggle.com/datasets/mobeenfatimah/diabetes-risk-prediction-dataset-50k-patients

El conjunto contiene 50.000 registros y 41 columnas. **Los registros son sintéticos, generados mediante reglas y procesos de simulación; no corresponden a pacientes reales ni a historias clínicas reales.** La etiqueta `Diabetes_Risk` representa las categorías de riesgo definidas por las reglas de generación del conjunto. Por esta razón, los resultados del proyecto son únicamente académicos y experimentales; no constituyen diagnóstico, tamizaje ni validación clínica.

## Partición de los datos

Responsable: Mafe

Se realizó la división del conjunto de datos en un 80% para entrenamiento y un 20% para prueba, utilizando train_test_split de Scikit-learn.

La partición se realizó de forma estratificada utilizando la variable objetivo Diabetes_Risk, con el propósito de conservar una proporción similar de las categorías Low, Moderate y High tanto en el conjunto de entrenamiento como en el conjunto de prueba.

Se utilizó random_state=42 para garantizar que la división sea reproducible y stratify=y para mantener la distribución de las clases.

Como resultado se obtuvieron:

Conjunto de entrenamiento: 40.000 registros (80%).
Conjunto de prueba: 10.000 registros (20%).

La división de los datos se realizó antes del ajuste del preprocesador. De esta manera, las transformaciones posteriores pueden ajustarse únicamente con los datos de entrenamiento y aplicarse posteriormente al conjunto de prueba, ayudando a prevenir Data Leakage.

El preprocesamiento de los datos, incluyendo la imputación de valores faltantes, codificación de variables categóricas y estandarización de variables numéricas, fue realizado posteriormente sobre los conjuntos obtenidos en esta etapa.

## Limpieza y Preprocesamiento de Datos

Responsable: Daniel

Después de realizar la partición de los datos en entrenamiento y prueba, se realizó el proceso de limpieza y preprocesamiento de las variables utilizadas para el modelo de Machine Learning.

### Diagnóstico de datos faltantes

El conjunto de datos contiene 50.000 registros y 41 columnas. Durante la revisión inicial se identificaron 20.810 valores faltantes distribuidos en 14 variables.

Como el dataset ya contenía valores faltantes, no fue necesario generar datos faltantes artificialmente.

No se encontraron registros duplicados.

### Selección de variables

Se utilizó Diabetes_Risk como variable objetivo.

Para las características predictoras se excluyeron las siguientes columnas:

Patient_ID
Diabetes_Risk_Score
AI_Health_Recommendation
Doctor_Consultation_Needed

Patient_ID corresponde a un identificador y no representa una característica predictiva.

Las demás variables excluidas pueden contener información relacionada con la evaluación o recomendación del riesgo, por lo que se decidió no utilizarlas como variables de entrada para evitar introducir información no deseada en el modelo.

Después de estas exclusiones se trabajó con 36 variables predictoras.

### Imputación de valores faltantes

Para las variables numéricas se utilizó imputación mediante la mediana:

SimpleImputer(strategy='median')

La mediana permite completar los valores faltantes sin verse afectada de la misma manera que la media por posibles valores atípicos.

Para las variables categóricas se utilizó imputación mediante la moda:

SimpleImputer(strategy='most_frequent')

De esta manera se reemplazaron los valores faltantes por la categoría que aparece con mayor frecuencia.

### Codificación de variables categóricas

Las variables categóricas fueron transformadas mediante One-Hot Encoding utilizando:

OneHotEncoder(handle_unknown='ignore', sparse_output=False)

El parámetro handle_unknown='ignore' permite procesar categorías que puedan aparecer posteriormente en los datos de prueba y que no hayan sido observadas durante el entrenamiento.

### Estandarización de variables numéricas

Las variables numéricas fueron estandarizadas utilizando:

StandardScaler()

Esto permite que las variables numéricas tengan una escala comparable, con una media cercana a 0 y una desviación estándar cercana a 1.

### Prevención de Data Leakage

El preprocesador se construyó utilizando ColumnTransformer.

Las transformaciones fueron ajustadas únicamente con X_train y posteriormente aplicadas a X_test.

De esta forma, las estadísticas utilizadas para la imputación y estandarización no se calculan utilizando información del conjunto de prueba.

### Resultado del preprocesamiento

Después de aplicar las transformaciones se obtuvieron:

X_train: 40.000 registros y 86 características.
X_test: 10.000 registros y 86 características.
Valores faltantes después del preprocesamiento en entrenamiento: 0.
Valores faltantes después del preprocesamiento en prueba: 0.

Los datos procesados quedaron preparados para la siguiente etapa del proyecto: el entrenamiento y evaluación de los modelos de Machine Learning.

### Archivos generados

Como resultado del proceso se generaron los siguientes archivos:

train_preprocesado.csv
test_preprocesado.csv
preprocesador_diabetes.pkl

El archivo preprocesador_diabetes.pkl contiene el preprocesador ajustado con los datos de entrenamiento y permite aplicar posteriormente las mismas transformaciones a nuevos datos.


## Análisis Exploratorio de Datos (EDA)

**Responsable: Daniel**

El análisis exploratorio se realizó sobre los 50.000 registros originales. No se encontraron filas duplicadas y se identificaron 20.810 valores faltantes distribuidos en 14 variables.

### Distribución de la variable objetivo

| Categoría | Registros | Porcentaje |
|---|---:|---:|
| High | 36.593 | 73,19% |
| Moderate | 12.937 | 25,87% |
| Low | 470 | 0,94% |

La clase `Low` representa menos del 1% del conjunto, por lo que existe un desbalance importante. Por esta razón se utiliza F1-macro y se revisan los resultados por clase, en lugar de interpretar únicamente accuracy.

### Hallazgos exploratorios

- Las medianas de `Fasting_Blood_Sugar` son 84,25 para Low, 100,60 para Moderate y 155,00 para High.
- También se observan medianas mayores de `HbA1c`, `Age` y `BMI` en High frente a Low.
- Al codificar la etiqueta como ordinal (Low=0, Moderate=1, High=2), las correlaciones lineales más altas con los predictores numéricos corresponden a `Fasting_Blood_Sugar` (0,380), `HbA1c` (0,275), `Age` (0,211) y `BMI` (0,161).
- La tasa de celdas faltantes en las 14 variables afectadas es similar entre las clases: 2,90% en Low, 2,99% en Moderate y 2,97% en High. En Low se identifican 191 celdas faltantes de 6.580 posibles en esas variables. Aunque la tasa no es superior, la clase tiene solo 470 registros, por lo que cada caso representa una proporción mayor de esa clase.

Estos hallazgos describen patrones del dataset sintético y no deben interpretarse como relaciones causales ni como evidencia clínica. Las gráficas de distribuciones, comparaciones por categoría, correlaciones y la tabla detallada de faltantes por clase se encuentran en el notebook `fase-1/notebooks/01_diabetes_risk_pipeline.ipynb`.

## Modelado y Evaluación
**Responsable:** Salo

Después de la partición y el preprocesamiento de los datos, se procedió a la construcción, entrenamiento y evaluación de los modelos de clasificación.

### Modelo base (baseline)

Antes de desarrollar el modelo principal se construyó un modelo base utilizando `DummyClassifier` con la estrategia `most_frequent`, el cual predice siempre la clase más frecuente del conjunto de entrenamiento (High). El propósito de este modelo es contar con un punto de comparación mínimo que permita determinar si el modelo principal realmente aporta una mejora.

### Modelo principal: Random Forest

Como modelo principal se seleccionó `RandomForestClassifier`, configurado con 100 árboles de decisión (`n_estimators=100`) y `random_state=42` para garantizar la reproducibilidad de los resultados. Random Forest construye múltiples árboles entrenados con distintas muestras de los datos y combina sus resultados mediante votación para generar la predicción final.

La comparación entre `class_weight=None` y `class_weight='balanced'` se incorporó como experimento reproducible en el script `fase-2/train.py`, mediante validación cruzada sobre el conjunto de entrenamiento. La prueba final se reserva para la evaluación de la configuración seleccionada.

### Métrica de evaluación

Se utilizó **F1-Score Macro** como métrica principal, complementada con Accuracy y la matriz de confusión. Se priorizó F1 macro sobre accuracy debido a que el problema es de clasificación multiclase con un desbalance considerable entre las categorías de riesgo, por lo que accuracy puede ocultar un desempeño deficiente en las clases minoritarias.

### Resultados

| Métrica | Baseline | Random Forest |
|---|---|---|
| Accuracy | 0.7319 | 0.8966 |
| F1-Score Macro | 0.2817 | 0.5699 |
| Mejora en F1 macro | | **+0.2882** |

La variable objetivo presenta un desbalance considerable: la clase **Low** representa menos del 1% de los registros de entrenamiento (376 de 40.000), frente a High (29.274) y Moderate (10.350). Esto explica que tanto el baseline como el Random Forest no logren identificar correctamente ningún caso de la clase Low, lo cual afecta directamente el valor del F1 macro obtenido.

### Prevención de fuga de información

El modelo se entrenó utilizando únicamente los datos de entrenamiento ya preprocesados por el equipo. El preprocesador, previamente ajustado solo con `X_train`, se integró junto con el modelo entrenado dentro de un mismo `Pipeline`, de forma que las transformaciones aplicadas a nuevos datos sean exactamente las mismas que se ajustaron durante el entrenamiento, sin recalcular ninguna estadística sobre el conjunto de prueba.

### Almacenamiento del modelo

El modelo final se almacenó integrando el preprocesador y el clasificador entrenado dentro de un único `Pipeline`, exportado mediante `joblib` (con compresión, para reducir el tamaño del archivo) en la ruta `fase-1/model/modelo.joblib`. Se verificó que el modelo almacenado puede cargarse nuevamente y generar predicciones correctas utilizando datos sin preprocesar, confirmando que el artefacto está listo para su reutilización en las siguientes fases del proyecto.

### Archivos generados

- `fase-1/model/modelo.joblib`
