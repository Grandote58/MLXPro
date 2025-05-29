# 🌳 Práctica: Construcción y Evaluación de un Árbol de Decisión en Google Colab

### 🎯 Objetivo de la Práctica

- Comprender el funcionamiento de los árboles de decisión para clasificación.
- Aplicar el algoritmo utilizando un conjunto de datos real.
- Evaluar el rendimiento del modelo mediante métricas adecuadas.
- Visualizar el árbol de decisión para interpretar las decisiones del modelo.

### 📁 Paso 1: Preparación del Entorno

Primero, asegúrate de tener acceso a Google Colab. Luego, importa las bibliotecas necesarias:

```python
# Importar bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
```

### 🌸 Paso 2: Cargar el Conjunto de Datos

Puedes encontrar conjuntos de datos abiertos en [Kaggle](https://www.kaggle.com/datasets) o en el [Repositorio de UCI](https://archive.ics.uci.edu/ml/index.php).

Utilizaremos el conjunto de datos Iris, que contiene información sobre tres especies de flores:

```python
# Cargar el conjunto de datos Iris
iris = load_iris()
X = iris.data
y = iris.target
```

### 🔍 Paso 3: Exploración de los Datos

Es importante entender la estructura de los datos antes de construir el modelo:

```python
# Crear un DataFrame para visualizar los datos
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(y, iris.target_names)
df.head()
```

### ✂️ Paso 4: Dividir los Datos en Entrenamiento y Prueba

Dividimos los datos para evaluar el rendimiento del modelo en datos no vistos:

```python
# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### 🧠 Paso 5: Entrenar el Modelo de Árbol de Decisión

Entrenamos el modelo utilizando el conjunto de entrenamiento:

```python
# Crear y entrenar el clasificador de árbol de decisión
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X_train, y_train)
```

### 📈 Paso 6: Evaluar el Modelo

Evaluamos el rendimiento del modelo utilizando el conjunto de prueba:

```python
# Realizar predicciones en el conjunto de prueba
y_pred = clf.predict(X_test)

# Mostrar la matriz de confusión y el informe de clasificación
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nInforme de Clasificación:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

### 🌳 Paso 7: Visualizar el Árbol de Decisión

Visualizamos el árbol para interpretar las decisiones del modelo:

```python
# Visualizar el árbol de decisión
plt.figure(figsize=(12,8))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Árbol de Decisión para Clasificación de Iris")
plt.show()
```

### 🧪 Paso 8: Experimentación Adicional

Para profundizar en la comprensión del modelo, puedes experimentar con los siguientes aspectos:

- Cambiar el criterio de división a 'entropy' para utilizar la ganancia de información.
- Ajustar la profundidad máxima del árbol (`max_depth`) y observar cómo afecta al rendimiento.
- Utilizar otros conjuntos de datos disponibles en `sklearn.datasets`.

### 📚 Recursos Adicionales

Para ampliar tus conocimientos sobre árboles de decisión y su implementación en Python, puedes consultar los siguientes recursos:

- Scikit-learn: Árboles de Decisión
- Artículo en Medium sobre Árboles de Decisión