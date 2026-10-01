# Diagnóstico, Limpieza y Preprocesamiento de Datos

## 1. Objetivo

Realizar el diagnóstico, limpieza y preprocesamiento del conjunto de datos utilizado para el proyecto de predicción del riesgo de diabetes, dejando los datos preparados para la etapa de entrenamiento y evaluación de los modelos de Machine Learning.

El proceso incluye la identificación de valores faltantes, revisión de duplicados, selección de variables predictoras, imputación de datos faltantes, codificación de variables categóricas y escalamiento de variables numéricas.

## 2. Diagnóstico inicial de los datos

El conjunto de datos contiene:

- 50.000 registros.
- 41 columnas.
- Variable objetivo: `Diabetes_Risk`.
- No se encontraron filas duplicadas.

Se identificaron 20.810 valores faltantes distribuidos en 14 variables.

### Valores faltantes encontrados

| Variable | Valores faltantes | Porcentaje |
|---|---:|---:|
| Weight_kg | 2.962 | 5,92% |
| Height_cm | 2.945 | 5,89% |
| Sleep_Hours | 2.017 | 4,03% |
| Exercise_Hours_Per_Week | 1.992 | 3,98% |
| HbA1c | 1.974 | 3,95% |
| Daily_Walking_Minutes | 1.964 | 3,93% |
| HDL | 1.000 | 2,00% |
| LDL | 1.000 | 2,00% |
| Triglycerides | 1.000 | 2,00% |
| Medication_Adherence | 1.000 | 2,00% |
| Total_Cholesterol | 1.000 | 2,00% |
| Blood_Glucose | 959 | 1,92% |
| Physical_Activity_Level | 500 | 1,00% |
| Age | 497 | 0,99% |

## 3. Distribución de la variable objetivo

La variable objetivo utilizada para la clasificación es `Diabetes_Risk`.

Su distribución es:

| Categoría | Cantidad | Porcentaje |
|---|---:|---:|
| High | 36.593 | 73,19% |
| Moderate | 12.937 | 25,87% |
| Low | 470 | 0,94% |

Esta distribución muestra que las clases no tienen la misma cantidad de observaciones, por lo que se conserva la distribución de las clases al realizar la división entre entrenamiento y prueba mediante `stratify`.

## 4. Selección de variables

Antes del preprocesamiento se separó la variable objetivo:

`Diabetes_Risk`

También se excluyeron las siguientes variables de las características utilizadas para el modelo:

- `Patient_ID`
- `Diabetes_Risk_Score`
- `AI_Health_Recommendation`
- `Doctor_Consultation_Needed`

`Patient_ID` se excluye porque funciona como identificador y no representa una característica predictiva.

Las otras tres variables se excluyen porque pueden contener información relacionada con la evaluación o recomendación asociada al riesgo de diabetes, lo que podría introducir información no deseada en el entrenamiento del modelo.

Después de estas exclusiones se trabajó con 36 variables predictoras.

## 5. División de los datos

Los datos se dividieron en:

- 80% para entrenamiento.
- 20% para prueba.

Con `random_state=42` y `stratify=y`.

Resultado:

- `X_train`: 40.000 registros.
- `X_test`: 10.000 registros.

La división se realizó antes de ajustar el preprocesador para evitar fuga de información (data leakage).

## 6. Identificación de variables

### Variables numéricas

Se identificaron 20 variables numéricas:

- `Age`
- `Height_cm`
- `Weight_kg`
- `BMI`
- `Waist_Circumference_cm`
- `Blood_Glucose`
- `HbA1c`
- `Fasting_Blood_Sugar`
- `Insulin_Level`
- `Blood_Pressure_Systolic`
- `Blood_Pressure_Diastolic`
- `Total_Cholesterol`
- `HDL`
- `LDL`
- `Triglycerides`
- `Heart_Rate`
- `Exercise_Hours_Per_Week`
- `Daily_Walking_Minutes`
- `Sleep_Hours`
- `Daily_Water_Intake_L`

### Variables categóricas

Se identificaron 16 variables categóricas:

- `Gender`
- `Country`
- `Physical_Activity_Level`
- `Diet_Quality`
- `Sugar_Intake_Level`
- `Stress_Level`
- `Smoking_Status`
- `Alcohol_Consumption`
- `Family_History_Diabetes`
- `Hypertension`
- `Heart_Disease`
- `Fatty_Liver`
- `PCOS`
- `Medication_Adherence`
- `Work_Type`
- `Residence_Type`

## 7. Imputación y transformación

Para las variables numéricas se utilizó:

- `SimpleImputer(strategy='median')`
- `StandardScaler()`

La mediana se utilizó para reemplazar los valores faltantes y posteriormente las variables numéricas fueron estandarizadas.

Para las variables categóricas se utilizó:

- `SimpleImputer(strategy='most_frequent')`
- `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`

La moda se utilizó para completar los valores faltantes de las variables categóricas y posteriormente se aplicó One-Hot Encoding.

## 8. Preprocesador

Las transformaciones se organizaron mediante `ColumnTransformer`, permitiendo aplicar un proceso diferente a las variables numéricas y categóricas.

El preprocesador se ajustó únicamente con los datos de entrenamiento:

`X_train`

Posteriormente, el mismo preprocesador se utilizó para transformar:

`X_test`

De esta forma se evita utilizar información del conjunto de prueba durante el ajuste de las transformaciones.

## 9. Resultados del preprocesamiento

Después de realizar la imputación, codificación y escalamiento se obtuvieron:

- `X_train` procesado: 40.000 registros y 86 características.
- `X_test` procesado: 10.000 registros y 86 características.
- Valores NaN después del preprocesamiento en entrenamiento: 0.
- Valores NaN después del preprocesamiento en prueba: 0.

Como comprobación del escalamiento, las variables numéricas presentaron medias cercanas a 0 y desviaciones estándar cercanas a 1.

## 10. Archivos generados

Como resultado del proceso se generaron los siguientes archivos:

- `train_preprocesado.csv`
- `test_preprocesado.csv`
- `preprocesador_diabetes.pkl`

`train_preprocesado.csv` contiene los datos de entrenamiento después del preprocesamiento.

`test_preprocesado.csv` contiene los datos de prueba después del preprocesamiento.

`preprocesador_diabetes.pkl` contiene el `ColumnTransformer` ajustado con los datos de entrenamiento para poder aplicar las mismas transformaciones posteriormente.



