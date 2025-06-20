# **"Cazadores de Hogares: Prediciendo el Precio de Casas con ML 🧠📈"**

### 🎯 **Objetivo**

Aplicar regresión lineal sobre un conjunto de datos reales (California Housing) para predecir el precio medio de una vivienda con base en variables como número de habitaciones, población, edad media del inmueble, entre otras.

### 📦 **Dataset**

Se utilizará el dataset público `California Housing` disponible en `sklearn.datasets`.
 ➡️ **Más info:** https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

### 🧪 **Paso a paso**

#### 🔹 Paso 1: Cargar librerías necesarias 📚

```python
# Manipulación de datos y visualización
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

#### 🔹 Paso 2: Cargar el dataset 🏡

```python
# Cargar dataset de California Housing
california = fetch_california_housing(as_frame=True)
df = california.frame

# Mostrar primeras filas
df.head()
```

#### 🔹 Paso 3: Visualizar datos importantes 📊

```python
# Correlación de variables
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("📌 Matriz de correlación entre variables")
plt.show()
```

#### 🔹 Paso 4: Dividir en conjunto de entrenamiento y prueba 📂

```python
# Separar variables predictoras y variable objetivo
X = df.drop(columns='MedHouseVal')  # Variables independientes
y = df['MedHouseVal']               # Variable dependiente (precio medio)

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### 🔹 Paso 5: Entrenar el modelo de regresión lineal 🤖

```python
# Crear modelo
modelo = LinearRegression()

# Entrenar con los datos de entrenamiento
modelo.fit(X_train, y_train)
```

#### 🔹 Paso 6: Realizar predicciones 🔮

```python
# Predicciones con datos de prueba
y_pred = modelo.predict(X_test)
```

#### 🔹 Paso 7: Evaluar el modelo 📏

```python
# Métricas de evaluación
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📉 Error Cuadrático Medio (MSE): {mse:.4f}")
print(f"📈 Coeficiente de Determinación (R²): {r2:.4f}")
```

#### 🔹 Paso 8: Visualizar resultados 🎨

```python
# Gráfico: valor real vs valor predicho
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Valor Real de la Vivienda 🏘️")
plt.ylabel("Valor Predicho 🧠")
plt.title("Comparación entre precios reales y predichos")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()
```

### ✅ **Reflexión final**

> El modelo de regresión lineal es una herramienta sencilla pero poderosa para establecer relaciones entre variables numéricas. En este caso, nos permite **predecir el valor medio de una casa** a partir de distintas características. 🏠💡

### 💡 Tips:

- Usa `StandardScaler` si hay mucha variación entre escalas de variables.
- Evalúa agregar o quitar variables y observar cómo cambia el R².
- Experimenta con otros modelos como Ridge o Lasso para evitar sobreajuste.