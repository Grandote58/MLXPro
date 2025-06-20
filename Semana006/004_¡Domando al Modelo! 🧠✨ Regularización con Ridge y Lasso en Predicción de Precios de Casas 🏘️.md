# **¡Domando al Modelo! 🧠✨ Regularización con Ridge y Lasso en Predicción de Precios de Casas 🏘️**

### 🎯 **Objetivo**

Entrenar y comparar modelos de regresión con **Ridge (L2)** y **Lasso (L1)** usando el dataset real **California Housing**, para identificar cómo la regularización evita el sobreajuste y mejora la generalización en problemas de predicción.

### 📦 **Dataset**

Utilizaremos el dataset de viviendas de California, disponible en `sklearn.datasets`.

🔗 [URL oficial de documentación](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)

### 🧪 **Paso a paso **

#### 🔹 Paso 1: Importar librerías necesarias 📚

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
```

#### 🔹 Paso 2: Cargar el dataset y visualizar 🏡

```python
# Cargar el dataset
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# Visualizar las primeras filas
X.head()
```

#### 🔹 Paso 3: Escalar características para la regresión ⚖️

```python
# Escalado de los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

#### 🔹 Paso 4: Dividir en entrenamiento y prueba ✂️

```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

#### 🔹 Paso 5: Entrenar modelos Ridge y Lasso 🤖

```python
# Crear modelos con valores estándar de alpha
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)

# Entrenar modelos
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)
```

#### 🔹 Paso 6: Realizar predicciones 🔮

```python
y_pred_ridge = ridge.predict(X_test)
y_pred_lasso = lasso.predict(X_test)
```

#### 🔹 Paso 7: Evaluar modelos con métricas 📏

```python
# Calcular métricas
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

# Mostrar resultados
print(f"📘 Ridge - MSE: {mse_ridge:.4f}, R²: {r2_ridge:.4f}")
print(f"📕 Lasso - MSE: {mse_lasso:.4f}, R²: {r2_lasso:.4f}")
```

#### 🔹 Paso 8: Visualizar y comparar 📊

```python
# Gráfica comparativa de predicción vs real
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_ridge, color='blue', alpha=0.5, label='Ridge')
plt.scatter(y_test, y_pred_lasso, color='green', alpha=0.5, label='Lasso')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal')
plt.xlabel("Precio real 🏷️")
plt.ylabel("Precio predicho 💸")
plt.title("Comparación de predicciones: Ridge vs Lasso")
plt.legend()
plt.grid(True)
plt.show()
```

### 💬 **Explicación : ¿Por qué usar Ridge y Lasso?**

📌 A veces, los modelos ajustan demasiado bien los datos de entrenamiento, memorizando patrones ruidosos. Esto es **overfitting**. La **regularización actúa como freno** para evitarlo.

🧰 Ridge:

- Penaliza coeficientes grandes.
- Útil cuando **todas las variables aportan algo**.
- No elimina variables, solo reduce su impacto.

🧹 Lasso:

- Puede llevar coeficientes a **cero**.
- Útil cuando **algunas variables son irrelevantes**.
- Ayuda a seleccionar las variables más importantes.

✔️ Ambos modelos te permiten lograr un balance entre **precisión y simplicidad**, lo cual es esencial para sistemas que deben generalizar bien.

### ✅ **Reflexión final**

> Elegir entre Ridge y Lasso depende de tu objetivo:
>
> - ¿Quieres un modelo más simple con menos variables? 👉 Usa **Lasso**.
> - ¿Prefieres mantener todas las variables pero reducir el sobreajuste? 👉 Usa **Ridge**.
>
> ¡La regularización es tu aliada para construir modelos robustos, claros y eficientes! 💪🤖

### 💡 Tips :

- 🔍 Prueba diferentes valores de `alpha` y observa su efecto.
- 📉 Usa validación cruzada para elegir el `alpha` óptimo (`GridSearchCV`).
- 🔢 Explora qué coeficientes se eliminan con Lasso: ¡te sorprenderás!