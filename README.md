# diabetes-risk-prediction
Proyecto de Machine Learning para predecir el nivel de riesgo de diabetes a partir de factores clínicos, demográficos y de estilo de vida.

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
