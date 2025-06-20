# Predicci-n-de-precios-de-Bitcoin-mediante-aprendizaje-autom-tico-en-Python
Predicción de precios de Bitcoin mediante aprendizaje automático en Python

![image](https://github.com/user-attachments/assets/2cfe1426-1289-4cf4-9103-45a934b8054e)

El aprendizaje automático resulta inmensamente útil en muchas industrias para automatizar tareas que antes requerían mano de obra humana; una de esas aplicaciones de ML es predecir si un comercio en particular será rentable o no.

En este artículo, aprenderemos cómo predecir una señal que indica si comprar una acción en particular será útil o no mediante el uso de ML.

Comencemos importando algunas bibliotecas que se utilizarán para varios propósitos, que se explicarán más adelante en este artículo.

Importación de bibliotecas
Las librerías de Python nos facilitan mucho el manejo de los datos y la realización de tareas típicas y complejas con una sola línea de código.

Pandas: esta biblioteca ayuda a cargar el marco de datos en un formato de matriz 2D y tiene múltiples funciones para realizar tareas de análisis de una sola vez.
Numpy: las matrices Numpy son muy rápidas y pueden realizar grandes cálculos en muy poco tiempo.
Matplotlib/Seaborn - Esta biblioteca se utiliza para dibujar visualizaciones.
Sklearn: este módulo contiene múltiples bibliotecas con funciones preimplementadas para realizar tareas desde el preprocesamiento de datos hasta el desarrollo y la evaluación de modelos.
XGBoost - Contiene el algoritmo de aprendizaje automático eXtreme Gradient Boosting, que es uno de los algoritmos que nos ayuda a lograr una alta precisión en las predicciones.

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sn

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn import metrics

import warnings

warnings.filterwarnings('ignore')

Importación de conjunto de datos

El conjunto de datos que utilizaremos aquí para realizar el análisis y construir un modelo predictivo son los datos de precios de Bitcoin. Utilizaremos los datos de OHLC ('Apertura', 'Máximo', 'Mínimo', 'Cierre') desde el 17 de julio de 2014 hasta el 29 de diciembre de 2022, es decir, durante 8 años para el precio de Bitcoin.

df = pd.read_csv('bitcoin.csv')

df.head()

Salida:
![image](https://github.com/user-attachments/assets/0917bdd0-4030-47e4-b87a-0af5094c8d66)


Primeras cinco filas de los datos

df.shape
Salida:

(2713, 7)
A partir de esto, llegamos a saber que hay 2904 filas de datos disponibles y para cada fila, tenemos 7 características o columnas diferentes.

df.describe()
Salida:
![image](https://github.com/user-attachments/assets/d8fbf9f8-20d7-4777-ad8b-10b9b3e52be5)

df.info()
Salida:
![image](https://github.com/user-attachments/assets/09d42db0-90f5-4cc0-aad5-cdb5482fd14b)

Análisis exploratorio de datos
EDA es un enfoque para analizar los datos mediante técnicas visuales. Se utiliza para descubrir tendencias y patrones, o para comprobar suposiciones con la ayuda de resúmenes estadísticos y representaciones gráficas.

Al realizar el EDA de los datos de precio de Bitcoin, analizaremos cómo se han movido los precios de la criptomoneda a lo largo del tiempo y cómo afecta el final de los trimestres a los precios de la moneda.


plt.figure(figsize=(15, 5))

plt.plot(df['Close'])

plt.title('Bitcoin Close price.', fontsize=15)

plt.ylabel('Price in dollars.')

plt.show()

Salida:

![image](https://github.com/user-attachments/assets/f7e4a866-3e0c-4c90-971f-fa636e710fd8)

Los precios de las acciones de Bitcoin muestran una tendencia alcista, como se muestra en el gráfico del precio de cierre de las acciones.


df[df['Close'] == df['Adj Close']].shape, df.shape
Salida:

((2713, 7), (2713, 7))

A partir de aquí podemos concluir que todas las filas de columnas 'Close' y 'Adj Close' tienen los mismos datos. Por lo tanto, tener datos redundantes en el conjunto de datos no va a ayudar, por lo que eliminaremos esta columna antes de un análisis más detallado.


df = df.drop(['Adj Close'], axis=1)

Ahora dibujemos el gráfico de distribución para las características continuas dadas en el conjunto de datos, pero antes de continuar, verifiquemos los valores nulos, si los hay, están presentes en el marco de datos.


df.isnull().sum()
Salida:

![image](https://github.com/user-attachments/assets/f9a1e666-f486-465a-8d5b-095b8046cb24)


Esto implica que no hay valores nulos en el conjunto de datos proporcionado.


features = ['Open', 'High', 'Low', 'Close']

plt.subplots(figsize=(20,10))

for i, col in enumerate(features):

  plt.subplot(2,2,i+1)
  
  sn.distplot(df[col])
  
plt.show()

salida:
![image](https://github.com/user-attachments/assets/cb4f22db-5acd-4ac3-8774-e236fc2b439d)

plt.subplots(figsize=(20,10))

for i, col in enumerate(features):

  plt.subplot(2,2,i+1)
  
  sn.boxplot(df[col], orient='h')
  
plt.show()

salida:

![image](https://github.com/user-attachments/assets/d393465c-9316-4c8a-ab39-927e7c7bb5ad)

Hay tantos valores atípicos en los datos, lo que significa que los precios de las acciones han variado enormemente en un período de tiempo muy corto. Comprobemos esto con la ayuda de un gráfico de barras.

Ingeniería de características
La ingeniería de características ayuda a derivar algunas características valiosas de las existentes. Estas características adicionales a veces ayudan a aumentar significativamente el rendimiento del modelo y, sin duda, ayudan a obtener información más profunda sobre los datos.

splitted = df['Date'].str.split('-', expand=True)

df['year'] = splitted[0].astype('int')
df['month'] = splitted[1].astype('int')
df['day'] = splitted[2].astype('int')

# Convierte la columna 'Fecha' en objetos datetime
df['Date'] = pd.to_datetime(df['Date']) 

df.head()

![image](https://github.com/user-attachments/assets/19b3226a-1db1-4a1c-8029-fb87847b469a)

Ahora tenemos tres columnas más, a saber, 'día', 'mes' y 'año', todas estas tres se han derivado de la columna 'Fecha' que se proporcionó inicialmente en los datos.


data_grouped = df.groupby('year').mean()

plt.subplots(figsize=(20,10))

for i, col in enumerate(['Open', 'High', 'Low', 'Close']):

  plt.subplot(2,2,i+1)
  
  data_grouped[col].plot.bar()
  
plt.show()

salida:
![image](https://github.com/user-attachments/assets/44ec156f-3e0c-40ae-ad35-7fa58bd9305e)

Aquí podemos observar por qué hay tantos valores atípicos en los datos, ya que los precios del bitcoin se han disparado en el año 2021.


df['is_quarter_end'] = np.where(df['month']%3==0,1,0)
df.head()
Salida:

![image](https://github.com/user-attachments/assets/48d60c7e-0a48-47b7-9210-5070613f149c)


df['open-close']  = df['Open'] - df['Close']

df['low-high']  = df['Low'] - df['High']

df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

Arriba hemos añadido algunas columnas más que ayudarán en el entrenamiento de nuestro modelo. Hemos agregado la función de objetivo, que es una señal de si comprar o no, entrenaremos nuestro modelo para predecir solo esto. Pero antes de continuar, verifiquemos si el objetivo está equilibrado o no utilizando un gráfico circular.

Arriba hemos añadido algunas columnas más que ayudarán en el entrenamiento de nuestro modelo. Hemos agregado la función de objetivo, que es una señal de si comprar o no, entrenaremos nuestro modelo para predecir solo esto. Pero antes de continuar, verifiquemos si el objetivo está equilibrado o no utilizando un gráfico circular.

plt.pie(df['target'].value_counts().values, 
        labels=[0, 1], autopct='%1.1f%%')
plt.show()

salida:

![image](https://github.com/user-attachments/assets/feeecd86-48df-477d-869a-a92918c0c2c4)

Cuando añadimos características a nuestro conjunto de datos, tenemos que asegurarnos de que no haya características altamente correlacionadas, ya que no ayudan en el proceso de aprendizaje del algoritmo.


plt.figure(figsize=(10, 10))

sn.heatmap(df.corr() > 0.9, annot=True, cbar=False)
plt.show()

salida:
![image](https://github.com/user-attachments/assets/0e6f7381-e369-4dce-a52b-31b133b9f8cf)

A partir del mapa de calor anterior, podemos decir que hay una alta correlación entre OHLC, lo cual es bastante obvio, y las características agregadas no están altamente correlacionadas entre sí o con las características proporcionadas anteriormente, lo que significa que estamos listos para comenzar a construir nuestro modelo.

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Suponiendo que df ya está definido
features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

# Escalando las características
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Divide los datos en conjuntos de entrenamiento y validación (prueba)
X_train, X_valid, Y_train, Y_valid = train_test_split(features, target, test_size=0.3, random_state=42)

# 'test_size=0.3' significa que el 30% de los datos se utilizarán para pruebas y el 70% para entrenamiento

Después de seleccionar las características en las que entrenar el modelo, debemos normalizar los datos, ya que los datos normalizados conducen a un entrenamiento estable y rápido del modelo. Después de eso, todos los datos se han dividido en dos partes con una proporción de 70/30 para que podamos evaluar el rendimiento de nuestro modelo en datos no vistos.

Desarrollo y evaluación de modelos
Ahora es el momento de entrenar algunos modelos de aprendizaje automático de última generación (Regresión logística, Máquina de vectores de soporte, XGBClassifier), y luego, en función de su rendimiento en los datos de entrenamiento y validación, elegiremos qué modelo de ML cumple mejor el propósito en cuestión.

Para la métrica de evaluación, usaremos la curva ROC-AUC, pero la razón es porque en lugar de predecir la probabilidad dura que es 0 o 1, nos gustaría que predijera probabilidades blandas que son valores continuos entre 0 y 1. Y con probabilidades blandas, la curva ROC-AUC se usa generalmente para medir la precisión de las predicciones.


models = [LogisticRegression(), SVC(kernel='poly', probability=True), XGBClassifier()]

for i in range(3):
  models[i].fit(X_train, Y_train)

  print(f'{models[i]} : ')
  print('Training Accuracy : ', metrics.roc_auc_score(Y_train, models[i].predict_proba(X_train)[:,1]))
  print('Validation Accuracy : ', metrics.roc_auc_score(Y_valid, models[i].predict_proba(X_valid)[:,1]))
  print()


  salida:
  ![image](https://github.com/user-attachments/assets/f3b81b0a-b750-4bc8-beee-d08d01b090fe)

  Entre los tres modelos, hemos entrenado XGBClassifier tiene el rendimiento más alto, pero se poda hasta el sobreajuste ya que la diferencia entre la precisión del entrenamiento y la validación es demasiado alta. Pero en el caso de la Regresión Logística, este no es el caso.

Ahora vamos a trazar una matriz de confusión para los datos de validación.

from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(models[0], X_valid, Y_valid, cmap='Blues')
plt.show()


salida:

![image](https://github.com/user-attachments/assets/20042bda-f5c1-48ca-8ef6-5fc02bae93da)

Y con eso Podemos ver que nuestro modelo está funcionando bien.

















