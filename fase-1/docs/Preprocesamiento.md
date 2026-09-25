# Limpieza y Preprocesamiento de Datos
## Proyecto: Diabetes Risk Prediction

**Responsable:** Daniel Díaz Escobar  
**Área:** Limpieza, Preprocesamiento de Datos e Imputación
---
## 1. Introducción
Esta etapa del proyecto tiene como objetivo preparar el conjunto de datos de Diabetes Risk Prediction
para su posterior utilización en modelos de Machine Learning.

El preprocesamiento permite identificar problemas en los datos, tratar valores faltantes,
transformar variables categóricas, estandarizar variables numéricas y preparar los datos de entrenamiento y prueba.

---
## 2. Objetivo general

Realizar la limpieza y el preprocesamiento del conjunto de datos de Diabetes Risk Prediction,
garantizando que los datos queden en un formato adecuado para ser utilizados posteriormente por
los algoritmos de Machine Learning.

---

## 3. Objetivos específicos

- Cargar y revisar el conjunto de datos.
- Analizar las dimensiones y tipos de variables.
- Identificar valores faltantes.
- Verificar la existencia de registros duplicados.
- Identificar variables numéricas y categóricas.
- Separar la variable objetivo de las variables predictoras.
- Dividir los datos en entrenamiento y prueba.
- Codificar las variables categóricas.
- Estandarizar las variables numéricas.
- Evitar problemas de Data Leakage.
- Validar que los datos procesados no tengan valores faltantes.
- Guardar los datos procesados y el preprocesador para su utilización posterior.
---

## 4. Descripción del conjunto de datos

El dataset utilizado corresponde a un conjunto de datos denominado Diabetes Risk Prediction.

El conjunto contiene aproximadamente 50.000 registros y 41 variables relacionadas
con diferentes características de los pacientes y factores asociados al riesgo de diabetes.

Entre las variables presentes se encuentran:

- Patient_ID
- Age
- Gender
- Country
- Height_cm
- Weight_kg
- BMI
- Blood_Glucose
- HbA1c
- Total_Cholesterol
- HDL
- LDL
- Triglycerides
- Physical_Activity_Level
- Medication_Adherence
- Diabetes_Risk
...

La variable objetivo utilizada para el proyecto es `Diabetes_Risk`, que representa el nivel de riesgo de diabetes
y será utilizada posteriormente por los modelos de clasificación.

---

## 5. Carga del dataset

El archivo fue cargado en Google Colab y posteriormente se descomprimió para obtener el archivo CSV correspondiente.

Después de cargarlo se verificaron las dimensiones del dataset para comprobar que los registros
y las variables fueran los esperados.

También se visualizaron las primeras filas, los nombres de las columnas y los tipos de datos de cada variable.

---

## 6. Exploración inicial

Durante la exploración inicial se revisaron:

- Primeras filas del dataset.
- Número de registros.
- Número de variables.
- Nombres de las columnas.
- Tipos de datos.
- Estadísticas descriptivas.
- Valores faltantes.
- Registros duplicados.

Esta exploración permitió conocer la estructura del conjunto de datos antes de realizar las transformaciones.

---

## 7. Análisis de valores faltantes

Se realizó un análisis mediante las funciones de Pandas para identificar la cantidad y el porcentaje de valores faltantes
presentes en cada columna.

Algunos de los valores faltantes encontrados fueron:

- Weight_kg: 5,92%.
- Height_cm: 5,89%.
- Sleep_Hours: 4,03%.
- HbA1c: aproximadamente 3,95%.
- Exercise_Hours_Per_Week: 3,98%.
- Daily_Walking_Minutes: 3,93%.
- Total_Cholesterol: 2%.
- HDL: 2%.
- LDL: 2%.
- Triglycerides: 2%.
- Medication_Adherence: 2%.
- Blood_Glucose: 1,92%.
- Age: 0,99%.
- Physical_Activity_Level: 1%.

Debido a que el dataset ya presentaba datos faltantes, se decidió trabajar directamente con ellos
y realizar una imputación.

---

## 8. Decisión sobre los datos faltantes


Durante el diagnóstico se comprobó que el dataset ya contiene valores faltantes en diferentes variables.


La decisión permite conservar los datos originales y aplicar directamente técnicas de imputación sobre los valores
ausentes existentes.
- **Variables numéricas:** se utilizó la **mediana** mediante `SimpleImputer(strategy="median")`. Los valores faltantes fueron
- reemplazados por la mediana calculada a partir de los datos de entrenamiento.

- **Variables categóricas:** se utilizó la **moda o categoría más frecuente** mediante `SimpleImputer(strategy="most_frequent")`.
- Los valores faltantes fueron reemplazados por la categoría que aparece con mayor frecuencia.

Estas técnicas permitieron conservar los registros del dataset sin eliminar las filas que contenían valores faltantes.

---

## 9. Verificación de registros duplicados

También se verificó la existencia de registros completamente duplicados mediante:

`df.duplicated().sum()`

Esta revisión permite determinar si existen filas repetidas que puedan afectar posteriormente el entrenamiento
de los modelos.

---

## 10. Variable objetivo

La variable objetivo seleccionada para el proyecto es:

`Diabetes_Risk`

Esta variable será la que el modelo de Machine Learning deberá predecir.

Las demás variables se consideran características o variables predictoras.

Se realizó la separación entre:

- `X`: variables predictoras.
- `y`: variable objetivo.

---

## 11. Variables excluidas

Antes de comenzar el procesamiento se excluyeron algunas columnas que no se utilizarán como variables predictoras:

- `Patient_ID`
- `Diabetes_Risk_Score`
- `AI_Health_Recommendation`
- `Doctor_Consultation_Needed`

### Patient_ID

`Patient_ID` corresponde a un identificador del paciente.
Al tratarse de un identificador, no representa una característica útil para predecir el riesgo de diabetes.

### Diabetes_Risk_Score

Esta variable puede representar información directamente relacionada con el nivel de riesgo.
Utilizarla como predictor podría provocar que el modelo recibiera información demasiado cercana a la variable objetivo.

Por esta razón se decidió excluirla para reducir el riesgo de Data Leakage.

### AI_Health_Recommendation

Esta columna contiene recomendaciones generadas a partir de información relacionada con el riesgo de salud.
Se excluyó para evitar utilizar información derivada que podría estar relacionada directamente con la variable objetivo.

### Doctor_Consultation_Needed

Esta variable representa una decisión relacionada con la necesidad de consulta médica. 
También puede estar relacionada con el nivel de riesgo, por lo que fue excluida para evitar
utilizar información derivada del resultado que se pretende predecir.

---

## 12. Identificación de variables numéricas y categóricas

Después de separar la variable objetivo, se identificaron las variables según su tipo de dato.

### Variables numéricas

Las variables numéricas contienen valores cuantitativos y pueden utilizarse en operaciones matemáticas.

Ejemplos:

- Age
- Height_cm
- Weight_kg
- BMI
- Blood_Glucose
- HbA1c
- Total_Cholesterol
- HDL
- LDL
- Triglycerides

Estas variables serán sometidas a imputación y posteriormente a estandarización.

### Variables categóricas

Las variables categóricas representan categorías o características no numéricas.

Ejemplos:

- Gender
- Country
- Physical_Activity_Level
- Medication_Adherence

Estas variables serán imputadas y posteriormente transformadas mediante One-Hot Encoding.

---

## 13. División de los datos

Antes de realizar el procesamiento se dividieron los datos en dos grupos:

- 80% para entrenamiento.
- 20% para prueba.

La división se realizó utilizando `train_test_split` de Scikit-learn.

Se utilizó:

- `test_size = 0.20`
- `random_state = 42`
- `stratify = y`

El parámetro `stratify` permite mantener una distribución similar de las clases de
la variable objetivo en los conjuntos de entrenamiento y prueba.

La división antes de ajustar los transformadores también permite evitar Data Leakage.

---

## 14. Imputación de variables numéricas

Para las variables numéricas se utilizó la estrategia de imputación mediante la mediana.

La mediana fue seleccionada porque es menos sensible a valores extremos que la media.

El procedimiento utilizado fue:

`SimpleImputer(strategy="median")`

Esto significa que cada valor faltante de una variable numérica será reemplazado por la mediana calculada
a partir de los datos de entrenamiento.

---

## 15. Imputación de variables categóricas

Para las variables categóricas se utilizó la categoría más frecuente.

La estrategia utilizada fue:

`SimpleImputer(strategy="most_frequent")`

Cuando una categoría está ausente, se reemplaza por la categoría que aparece con mayor
frecuencia dentro de los datos de entrenamiento.

---

## 16. Codificación de variables categóricas

Después de realizar la imputación, las variables categóricas fueron transformadas mediante One-Hot Encoding.

Se utilizó:

`OneHotEncoder(handle_unknown="ignore", sparse_output=False)`

El One-Hot Encoding convierte cada categoría en una variable binaria.

Por ejemplo, si una variable `Gender` contiene:

- Male
- Female

puede transformarse en columnas como:

- Gender_Male
- Gender_Female

Esto permite que los modelos de Machine Learning puedan trabajar con las categorías en formato numérico.

El parámetro `handle_unknown="ignore"` permite procesar categorías que puedan aparecer posteriormente en datos
nuevos y que no hayan estado presentes durante el entrenamiento.

---

## 17. Estandarización de variables numéricas

Después de realizar la imputación, las variables numéricas fueron estandarizadas mediante `StandardScaler`.

La estandarización transforma las variables para que tengan aproximadamente:

- Media = 0.
- Desviación estándar = 1.

Se utilizó:

`StandardScaler()`

Este procedimiento ayuda a que las variables numéricas estén en escalas comparables.

---

## 18. Uso de Pipelines

Para organizar el procesamiento se utilizaron Pipelines de Scikit-learn.

Se creó un pipeline para variables numéricas:

1. Imputación mediante mediana.
2. Estandarización mediante StandardScaler.

Para las variables categóricas:

1. Imputación mediante la categoría más frecuente.
2. One-Hot Encoding.

El uso de pipelines permite organizar las operaciones y asegurar que las transformaciones se ejecuten en el orden correcto.

---

## 19. Uso de ColumnTransformer

También se utilizó `ColumnTransformer` para aplicar diferentes procesos dependiendo del tipo de variable.

Las variables numéricas pasan por:

**Imputación por mediana → StandardScaler**

Las variables categóricas pasan por:

**Imputación por moda → One-Hot Encoding**

De esta manera, cada grupo de variables recibe el tratamiento adecuado.

---

## 20. Prevención de Data Leakage

Uno de los puntos importantes del preprocesamiento fue evitar Data Leakage.

El Data Leakage ocurre cuando información del conjunto de prueba termina influyendo en el procesamiento o entrenamiento del modelo.

Para evitarlo, primero se realizó la división entre entrenamiento y prueba.

Posteriormente:

- El preprocesador se ajustó únicamente con `X_train`.
- `X_train` se transformó utilizando `fit_transform()`.
- `X_test` solamente se transformó utilizando `transform()`.

De esta forma, las estadísticas utilizadas para calcular las medianas, las categorías y los parámetros de escalado provienen únicamente del conjunto de entrenamiento.

---

## 21. Aplicación del preprocesador

Para el conjunto de entrenamiento se utilizó:

`X_train_processed = preprocessor.fit_transform(X_train)`

Para el conjunto de prueba:

`X_test_processed = preprocessor.transform(X_test)`

La diferencia es importante porque el preprocesador aprende los parámetros únicamente a partir de los datos de entrenamiento.

---

## 22. Validación de valores faltantes

Después del procesamiento se verificó nuevamente la existencia de valores NaN.

Se comprobó tanto el conjunto de entrenamiento como el conjunto de prueba.

El objetivo es garantizar que después de la imputación no existan valores faltantes en las variables utilizadas por el modelo.

La validación se realizó utilizando:

`np.isnan(X_train_processed).sum()`

y:

`np.isnan(X_test_processed).sum()`

El resultado esperado es 0 en ambos conjuntos.

---

## 23. Obtención de nombres de variables

Después del One-Hot Encoding, algunas variables categóricas originales se convierten en varias columnas.

Por esta razón se obtuvieron los nombres finales de las variables utilizando:

`preprocessor.get_feature_names_out()`

Esto permite conocer exactamente cuáles son las variables resultantes después del procesamiento.

---

## 24. Creación de los datasets finales

Después de realizar todas las transformaciones se construyeron dos DataFrames finales:

- `train_final`
- `test_final`

Cada uno contiene las variables ya procesadas y la variable objetivo `Diabetes_Risk`.

El conjunto de entrenamiento se utilizará posteriormente para entrenar los modelos.

El conjunto de prueba se utilizará para evaluar el desempeño de los modelos con datos que no fueron utilizados durante el entrenamiento.

---

## 25. Validación final

Se realizó una validación final para comprobar:

- Cantidad de registros de entrenamiento.
- Cantidad de registros de prueba.
- Cantidad de valores faltantes.
- Cantidad de valores NaN.
- Número de variables finales.

La validación permite confirmar que el dataset está listo para pasar a la siguiente etapa del proyecto.

---

## 26. Archivos generados

Como resultado del proceso se generaron los siguientes archivos:

`train_preprocesado.csv`

Contiene los datos de entrenamiento después de realizar la imputación, codificación y estandarización.

`test_preprocesado.csv`

Contiene los datos de prueba después de aplicar las mismas transformaciones aprendidas durante el entrenamiento.

`preprocesador_diabetes.pkl`

Contiene el preprocesador utilizado para transformar los datos.

Guardar el preprocesador permite utilizar exactamente las mismas transformaciones posteriormente cuando se incorporen nuevos datos al sistema.

---

## 27. Tecnologías utilizadas

Para realizar esta etapa se utilizaron:

- Python
- Google Colab
- Pandas
- NumPy
- Scikit-learn
- Joblib

### Librerías principales

Pandas fue utilizada para la manipulación y análisis del dataset.

NumPy fue utilizada para operaciones numéricas y validaciones.

Scikit-learn fue utilizada para:

- División de datos.
- Imputación.
- Pipelines.
- ColumnTransformer.
- One-Hot Encoding.
- StandardScaler.

Joblib fue utilizada para guardar el preprocesador.

---

## 28. Resumen de decisiones

| Elemento | Decisión |
|---|---|
| Dataset | Diabetes Risk Prediction |
| Variable objetivo | Diabetes_Risk |
| Datos faltantes | Ya estaban presentes en el dataset |
| Datos faltantes artificiales | No se agregaron |
| Variables numéricas | Mediana + StandardScaler |
| Variables categóricas | Moda + One-Hot Encoding |
| División | 80% entrenamiento / 20% prueba |
| Random State | 42 |
| Estratificación | Sí |
| Identificador | Patient_ID excluido |
| Variables derivadas del riesgo | Excluidas para reducir Data Leakage |
| Preprocesamiento | Pipeline + ColumnTransformer |
| Validación | Comprobación de NaN y valores faltantes |
| Preprocesador | Guardado como archivo .pkl |

---

## 29. Resultado final

Después del proceso de limpieza y preprocesamiento, el dataset queda preparado para ser utilizado por los modelos de Machine Learning.

Los valores faltantes fueron tratados mediante técnicas de imputación, las variables categóricas fueron convertidas a formato numérico mediante One-Hot Encoding y las variables numéricas fueron estandarizadas.

Además, el procesamiento se realizó evitando Data Leakage, ya que los parámetros de las transformaciones fueron aprendidos únicamente a partir del conjunto de entrenamiento.

Finalmente, se generaron los archivos:

- `train_preprocesado.csv`
- `test_preprocesado.csv`
- `preprocesador_diabetes.pkl`

Estos archivos permiten continuar con la siguiente etapa del proyecto, correspondiente al entrenamiento y evaluación de los modelos de Machine Learning.

---

## 30. Conclusiones

El proceso de limpieza y preprocesamiento permitió transformar el dataset original en un conjunto de datos adecuado para su utilización en Machine Learning.

La revisión inicial permitió identificar los valores faltantes existentes y tomar la decisión de no introducir datos faltantes artificialmente.

Las variables numéricas fueron tratadas mediante imputación por mediana y posteriormente estandarizadas. Las variables categóricas fueron tratadas mediante imputación por la categoría más frecuente y One-Hot Encoding.

También se realizó una división de los datos en entrenamiento y prueba antes de ajustar el preprocesador, lo que ayuda a evitar Data Leakage.

Finalmente, los datos procesados fueron validados y almacenados junto con el preprocesador, dejando la información preparada para la fase de entrenamiento de los modelos.
