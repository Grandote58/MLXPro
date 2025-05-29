# 📘 Práctica: Ajuste Lineal por Mínimos Cuadrados

## 🎯 Objetivo:

- Aplicar la fórmula matemática de mínimos cuadrados para encontrar la recta de mejor ajuste.
- Implementar el modelo desde cero (sin librerías de regresión).
- Visualizar el resultado de forma clara y didáctica.

## 🧰 Paso 1: Importar librerías necesarias

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## 📥 Paso 2: Cargar datos desde repositorio abierto

Usamos un dataset simple de viviendas desde GitHub:

```python
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

# Usamos solo una variable: promedio de habitaciones vs. precio medio
X = df['rm'].values  # número medio de habitaciones
y = df['medv'].values  # precio medio de la vivienda

# Visualizar una muestra
df[['rm', 'medv']].head()
```

## ✍️ Paso 3: Implementar regresión por mínimos cuadrados desde cero

```python
# Cálculo de los coeficientes
n = len(X)
x_mean = np.mean(X)
y_mean = np.mean(y)

# Numerador y denominador para la pendiente (beta_1)
numerador = np.sum((X - x_mean) * (y - y_mean))
denominador = np.sum((X - x_mean)**2)
beta_1 = numerador / denominador

# Intercepto (beta_0)
beta_0 = y_mean - beta_1 * x_mean

# Función predictiva
y_pred = beta_0 + beta_1 * X

print(f"Intercepto (β0): {beta_0:.2f}")
print(f"Pendiente (β1): {beta_1:.2f}")
```

## 📊 Paso 4: Evaluar el modelo

```python
# Calcular el error cuadrático medio (MSE)
mse = np.mean((y - y_pred)**2)
print(f"Error cuadrático medio (MSE): {mse:.2f}")
```

## 📈 Paso 5: Visualizar el resultado

```python
# Gráfico de dispersión + línea de ajuste
plt.figure(figsize=(10,6))
plt.scatter(X, y, color='blue', alpha=0.5, label='Datos reales')
plt.plot(X, y_pred, color='red', linewidth=2, label='Recta de mínimos cuadrados')
plt.xlabel('Promedio de habitaciones (rm)')
plt.ylabel('Precio medio ($1000s)')
plt.title('Regresión Lineal por Mínimos Cuadrados')
plt.legend()
plt.grid(True)
plt.show()
```

## ✅ Conclusiones

- Aplicamos el método clásico de regresión por mínimos cuadrados sin usar librerías avanzadas.
- Interpretamos los coeficientes calculados manualmente.
- Visualizamos la recta de mejor ajuste sobre un conjunto real de datos.