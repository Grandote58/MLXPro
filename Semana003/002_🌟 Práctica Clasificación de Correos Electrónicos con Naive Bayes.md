# **🌟 Práctica: Clasificación de Correos Electrónicos con Naive Bayes**

## 🎯 Objetivo de la Práctica

- Comprender el funcionamiento del algoritmo Naive Bayes para clasificación de texto.
- Aplicar el algoritmo para clasificar correos electrónicos como spam o no spam.
- Evaluar el rendimiento del modelo utilizando métricas adecuadas.

## 🧰 Paso 1: Preparación del Entorno

Importamos las bibliotecas necesarias:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
```

## 📥 Paso 2: Cargar el Conjunto de Datos

Utilizaremos un conjunto de datos de correos electrónicos etiquetados como spam o no spam. Puedes encontrar conjuntos de datos abiertos en [Kaggle](https://www.kaggle.com/datasets) o en el [Repositorio de UCI](https://archive.ics.uci.edu/ml/index.php).

```python
# Cargar el conjunto de datos
# Asegúrate de que el archivo 'spam.csv' esté en tu entorno de trabajo
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
```

## 🔍 Paso 3: Exploración de los Datos

Echamos un vistazo a los datos:

```python
# Mostrar las primeras filas
df.head()

# Distribución de clases
df['label'].value_counts().plot(kind='bar', title='Distribución de Clases')
plt.show()
```

## ✂️ Paso 4: Preprocesamiento de los Datos

Convertimos las etiquetas a valores numéricos y dividimos el conjunto de datos:

```python
# Convertir etiquetas a valores numéricos
df['label_num'] = df.label.map({'ham':0, 'spam':1})

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label_num'], test_size=0.3, random_state=42)
```

## 🧠 Paso 5: Vectorización de Texto

Transformamos el texto en una representación numérica utilizando Bag of Words:

```python
# Inicializar el vectorizador
vectorizer = CountVectorizer()

# Ajustar y transformar los datos de entrenamiento
X_train_vec = vectorizer.fit_transform(X_train)

# Transformar los datos de prueba
X_test_vec = vectorizer.transform(X_test)
```

## 🏗️ Paso 6: Entrenar el Modelo Naive Bayes

Entrenamos el modelo utilizando el clasificador Naive Bayes Multinomial:

```python
# Inicializar el clasificador
nb = MultinomialNB()

# Entrenar el modelo
nb.fit(X_train_vec, y_train)
```

## 📈 Paso 7: Evaluar el Modelo

Evaluamos el rendimiento del modelo en el conjunto de prueba:

```python
# Realizar predicciones
y_pred = nb.predict(X_test_vec)

# Mostrar la matriz de confusión
conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Realidad')
plt.show()

# Mostrar el informe de clasificación
print(classification_report(y_test, y_pred))
```

## 🧪 Paso 8: Prueba con Nuevos Datos

Probamos el modelo con un nuevo mensaje:

```python
# Nuevo mensaje
new_message = ['¡Gana dinero rápido con este truco!']

# Transformar el mensaje
new_message_vec = vectorizer.transform(new_message)

# Predecir
prediction = nb.predict(new_message_vec)

# Mostrar resultado
print('Spam' if prediction[0] == 1 else 'No Spam')
```