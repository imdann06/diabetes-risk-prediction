
Entrenamiento del modelo — Clasificación de riesgo de diabetes

Descripción 

Para la parte de modelado dividimos el trabajo en dos celdas principales : la primera contiene el modelo baseline y la segunda el modelo principal.

Modelo baseline
El baseline se construyó para tener una referencia de qué tan bien se puede predecir sin que el modelo realmente aprenda nada, y así después poder comparar si el modelo principal sí aporta una mejora . Si el modelo principal no logra superar al baseline, es señal de que algo anda mal.

Se implementó usando DummyClassifier con la estrategia most_frequent, que simplemente predice siempre la clase que más se repite en los datos de entrenamiento que para  este caso es "High".

Después de entrenarlo, se hacen las predicciones sobre los datos de prueba y se calculan estas métricas:

Accuracy: el porcentaje de predicciones correctas.
F1-Score Macro: calcula el F1 de cada clase por separado y saca el promedio, dándole el mismo peso a las tres clases. La usamos como métrica principal porque, como el dataset está desbalanceado, el accuracy por sí solo puede verse bien aunque el modelo esté ignorando por completo una clase.
También se hace una matriz de confusión para ver en qué clases se equivoca más.

Al final se arma un pipeline que junta el preprocesador con el baseline. Aunque los datos que usamos ya venían preprocesados, meter el preprocesador dentro del pipeline sirve para poder usarlo después con datos nuevos sin procesar.

Resultado del baseline: Accuracy 0.7319 y F1-Score Macro 0.2817. El accuracy parece bueno a simple vista, pero el F1 macro  bajo muestra que en realidad el baseline solo acierta prediciendo siempre "High" y no distingue nada de las otras clases.

Modelo principal: Random Forest
Como grupo decidimos usar Random Forest porque ya teníamos algo de conocimiento previo del modelo y nos sentíamos más cómodos trabajando con él.

Random Forest arma varios árboles de decisión en nuestro caso se decidió por  100, cada uno entrenado con una muestra distinta de los datos. Para predecir, cada árbol vota por una clase y se elige la que tenga más votos.

Le calculamos las mismas métricas que al baseline: Accuracy, F1-Score Macro y matriz de confusión, para poder compararlos .

También probamos ponerle class_weight='balanced' para ver si mejoraba la detección de la clase "Low" la más chiquita, es decir, no hay suficientes ejemplos para que aprenda el patrón, pero el resultado salió un poco peor que sin ese parámetro, así que lo descartamos.


Resultados

Métrica	        Baseline	Random Forest
Accuracy	        0.7319	       0.8966
F1-Score Macro	  0.2817	       0.5699

Mejora en F1 macro		  +0.2882

La variable objetivo que es Diabetes_Risk en entrenamiento se reparte así: High = 29,274 , Moderate 10,350 , Low 376. Low es apenas el aproximadamente 1% de los datos, y por eso ni el baseline ni el Random Forest logran predecirla bien.

Interpretación de resultados
¿El modelo mejora al baseline? 
Sí, ya que pasamos de un F1 macro de 0.2817 a 0.5699, una mejora de +0.2882. Eso nos dice que el Random Forest sí está aprendiendo algo de los datos y no solo repitiendo la clase más común como hacía el baseline.

¿La métrica es razonable? 
Creemos que sí, tomando en cuenta que las clases están muy desbalanceadas. El modelo predice bastante bien "High" con F1 0.94 y Moderate con F1 0.77, pero no logra identificar ningún caso de "Low". Eso hace que el promedio macro baje, aunque el accuracy general 0.90 se vea alto.

¿Cuáles fueron las principales dificultades?
La más grande fue el desbalance tan fuerte de la clase "Low" solo 376 de 40,000 . Intentamos arreglarlo con class_weight='balanced', pero no ayudó , de hecho el F1 macro bajó un poco (0.5539 vs 0.5699), lo que nos hizo ver que el problema es que simplemente hay muy pocos ejemplos de este para que el modelo aprenda a reconocerla.

¿Qué se podría mejorar a futuro?

Se podría mejorar la detección de la clase "Low" probando otra técnica que pueda generar datos nuevos de esa clase para que el modelo tenga más ejemplos para aprender. También ayudaría conseguir más datos reales de "Low", ya que 376 ejemplos es muy poco. Otra opción es probar distintos parámetros del Random Forest con GridSearchCV, en vez de dejarlo con los valores por defecto. Y por último, se podrían probar otros modelos como XGBoost o LightGBM.

¿Cómo funciona el código?
El código carga los datos ya preprocesados y separa las variables de entrada de la variable que queremos predecir (Diabetes_Risk). Se entrena el modelo con los datos de entrenamiento y se generan las predicciones sobre los datos de prueba.

Después se calculan las métricas y se muestra la matriz de confusión para analizar los resultados. Por último, el modelo entrenado y el preprocesador se juntan en un pipeline que se exporta como archivo .joblib, para poder guardarlo y volver a usarlo sin tener que entrenarlo de nuevo.

Cómo ejecutar esta parte: 

train_preprocesado.csv, test_preprocesado.csv y preprocesador_diabetes.pkl son los archivos necesarios.
instalar instalado pandas, scikit-learn, joblib y matplotlib.

Pasos:
subir los archivos
Correr la celda que carga los datos y el preprocesador.
Correr la celda del baseline (entrena, saca métricas y muestra la matriz de confusión).
Correr la celda del Random Forest (entrena, saca métricas y muestra la matriz de confusión).
Correr la celda que arma y exporta el pipeline final en fase-1/modelo.joblib.
Opcional pero recomendado: Correr la celda de prueba / verificación, que carga de nuevo el modelo.joblib y hace predicciones sobre datos sin preprocesar, para confirmar que todo quedó bien guardado.
Con correr las celdas en orden indicado o como all run.
