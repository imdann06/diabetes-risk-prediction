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
