# **📘 Práctica: Predicción del Precio de Casas con Regresión Lineal**

### 🧠 Objetivo:

- Comprender el funcionamiento de la regresión lineal simple.
- Aplicar el modelo usando un conjunto de datos del mundo real.
- Visualizar y evaluar el desempeño del modelo.

## 🧰 Paso 1: Preparación del entorno

```python
# Importar bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
```

## 📥 Paso 2: Carga de datos desde repositorio open source

Usamos el dataset **BostonHousing** desde GitHub (por Selva Prabhakaran):

```python
# Cargar el dataset directamente desde GitHub
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)
df.head()
```

## 🔍 Paso 3: Selección de variables

Elegimos una variable predictora (`rm`: promedio de habitaciones por casa) y una respuesta (`medv`: valor medio de la vivienda en miles de USD).

```python
# Variables independientes y dependientes
X = df[['rm']]
y = df['medv']
```

## ✂️ Paso 4: División en entrenamiento y prueba

```python
# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

## 📈 Paso 5: Ajuste del modelo

```python
# Crear el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Mostrar coeficientes
print("Intercepto:", modelo.intercept_)
print("Coeficiente:", modelo.coef_[0])
```

## 📊 Paso 6: Predicción y evaluación

```python
# Realizar predicciones
y_pred = modelo.predict(X_test)

# Evaluación
print("Error cuadrático medio (MSE):", mean_squared_error(y_test, y_pred))
print("Coeficiente de determinación (R²):", r2_score(y_test, y_pred))
```

## 🧪 Paso 7: Visualización del modelo

Creamos un gráfico de dispersión con la recta ajustada de la regresión:

```python
# Visualización de resultados
plt.figure(figsize=(10,6))
plt.scatter(X_test, y_test, color='blue', label='Datos reales')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Modelo lineal')
plt.title('Regresión Lineal: Precio vs Número de habitaciones')
plt.xlabel('Número promedio de habitaciones (rm)')
plt.ylabel('Precio medio de la vivienda ($1000s)')
plt.legend()
plt.grid(True)
plt.show()
```

## ✅ Conclusión

- Aprendiste a cargar un dataset real desde un repositorio abierto.
- Aplicaste regresión lineal para predecir precios de vivienda.
- Interpretaste los coeficientes y visualizaste la recta de regresión.