# **🎯 Práctica: Diagnóstico de Enfermedad Cardiaca con Regresión Logística**

## 📘 Objetivo de la Práctica

- Comprender el funcionamiento matemático y computacional de la regresión logística.
- Aplicar el modelo a un conjunto de datos del mundo real.
- Evaluar su precisión y visualizar el resultado.

## 🧰 Paso 1: Preparación del Entorno

```python
# Librerías esenciales
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
```

## 📥 Paso 2: Cargar Datos desde Repositorio Abierto

Usamos el conjunto de datos de **enfermedad cardíaca** del repositorio [YBI Foundation en GitHub](https://github.com/ybifoundation/Dataset).

```python
# Cargar los datos
url = "https://raw.githubusercontent.com/ybifoundation/Dataset/main/Heart.csv"
df = pd.read_csv(url)

# Visualizar las primeras filas
df.head()
```

## 🔍 Paso 3: Exploración y Preparación de los Datos

```python
# Verificar tipos de datos y valores nulos
print(df.info())
print(df.isnull().sum())

# Ver distribución de la variable objetivo
df['HeartDisease'].value_counts().plot(kind='bar', title='Distribución de Clases')
plt.show()
```

## ✂️ Paso 4: División de Datos

```python
# Variables predictoras y objetivo
X = df.drop(columns='HeartDisease')
y = df['HeartDisease']

# División en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

## 🧠 Paso 5: Entrenamiento del Modelo

```python
# Crear y entrenar el modelo de regresión logística
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)
```

## 📊 Paso 6: Evaluación del Modelo

```python
# Predicciones
y_pred = modelo.predict(X_test)

# Evaluación
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))
```

## 📈 Paso 7: Visualización de Resultados

```python
# Visualización de la matriz de confusión
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="YlGnBu")
plt.title("Matriz de Confusión - Regresión Logística")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.show()
```

## ✅ Conclusiones de Aprendizaje

- La regresión logística permite modelar la probabilidad de eventos binarios.
- Es eficiente y rápida en problemas linealmente separables.
- La interpretación de coeficientes permite entender el impacto de cada variable en el resultado.