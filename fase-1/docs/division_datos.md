1. Partición Estratificada y Prevención de Data Leakage:

Se realiza una división estratificada del dataset (80% entrenamiento, 20% prueba) utilizando la variable objetivo Diabetes_Risk. La estratificación asegura que la    proporción de las clases (Low, Moderate, High) se mantenga intacta en ambos subconjuntos. Para garantizar la total ausencia de fuga de información (Data Leakage), todas las transformaciones de preprocesamiento se encadenan mediante un ColumnTransformer que ajusta sus parámetros (fit_transform) únicamente con X_train, aplicando posteriormente dicha transformación (transform) sobre X_test sin recalcular estadísticas.

2. Criterios para las Transformaciones Elegidas:

Imputación por Mediana (Variables Numéricas): Se opta por la mediana en lugar de la media para imputar los valores faltantes en las variables cuantitativas, debido a que la mediana es una medida de tendencia central robusta ante la presencia de datos atípicos (outliers) o distribuciones con sesgo.

Imputación por Moda (Variables Categóricas): Para las variables cualitativas con datos faltantes, se imputa con la moda (most frequent), preservando la categoría categórica de mayor prevalencia sin alterar la estructura cualitativa.

One-Hot Encoding (Variables Categóricas): Se transforma la información categórica no ordinal a una representación vectorial binaria (dummy). Se utiliza el parámetro handle_unknown='ignore' para prevenir errores en tiempo de inferencia ante categorías no observadas durante el entrenamiento en X_train.

Estandarización / StandardScaler (Variables Numéricas): Se escalan los atributos cuantitativos para que tengan una media de 0 y una desviación estándar de 1. Esto equipara la magnitud de todas las variables, previniendo que atributos con rangos numéricos elevados dominen de manera indebida sobre modelos sensibles a la escala (como regresión logística, máquinas de soporte vectorial o redes neuronales).
