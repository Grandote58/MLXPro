# **"¡Regulariza tu Barrio! 🏡🧠 Cómo Ridge y Lasso Predicen Mejor que Adivinar"**

### 🎯 **Objetivo**

Demostrar la **importancia de dividir adecuadamente los datos** en entrenamiento y prueba para evitar sobreajuste, mientras se comparan dos modelos de regresión regularizada: **Ridge** (L2) y **Lasso** (L1), usando el dataset California Housing.

### 📦 **Dataset**

Se usará el dataset **California Housing** disponible en `sklearn.datasets`.

🔗 **URL oficial**:
 https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

### 🧪 **Paso a paso**

#### 🔹 Paso 1: Cargar librerías 🧰

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
```

#### 🔹 Paso 2: Cargar y explorar el dataset 🏘️

```python
# Cargar los datos
california = fetch_california_housing(as_frame=True)
df = california.frame

# Visualizar las primeras filas
df.head()
```

#### 🔹 Paso 3: Dividir en variables y aplicar escalado ⚖️

```python
# Separar variables predictoras y objetivo
X = df.drop(columns='MedHouseVal')
y = df['MedHouseVal']

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

#### 🔹 Paso 4: División en Training/Test 🧠✂️

```python
# División 80/20
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Verificar tamaños
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
```

#### 🔹 Paso 5: Entrenar Ridge y Lasso 🤖

```python
# Crear y entrenar modelos
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)

ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)
```

#### 🔹 Paso 6: Hacer predicciones 🔮

```python
y_pred_ridge = ridge.predict(X_test)
y_pred_lasso = lasso.predict(X_test)
```

#### 🔹 Paso 7: Evaluar los modelos 📏

```python
# Métricas para Ridge
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

# Métricas para Lasso
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print(f"📘 Ridge - MSE: {mse_ridge:.4f}, R²: {r2_ridge:.4f}")
print(f"📕 Lasso - MSE: {mse_lasso:.4f}, R²: {r2_lasso:.4f}")
```

#### 🔹 Paso 8: Visualizar comparaciones 📊

```python
# Gráfica comparativa
plt.figure(figsize=(8, 6))
plt.plot(y_test.values, label='Valor Real', color='black')
plt.plot(y_pred_ridge, label='Ridge', alpha=0.7)
plt.plot(y_pred_lasso, label='Lasso', alpha=0.7)
plt.title("Comparación de predicciones 🏠")
plt.xlabel("Índice de muestra")
plt.ylabel("Precio medio")
plt.legend()
plt.show()
```

### 💬 **Explicación:  ¿Por qué dividir los datos?**

📌 Imagina que estás entrenando a un estudiante con ejemplos para un examen. Si luego le das el mismo examen que usaste en clase, no sabrás si **realmente aprendió o solo memorizó**.

🔍 Lo mismo pasa en Machine Learning:

- Si entrenas y pruebas con los **mismos datos**, el modelo se puede **"sobreajustar"**: aprende demasiado bien los ejemplos, pero falla en casos nuevos.
- Dividir los datos permite **medir con realismo la capacidad del modelo para generalizar** 🧠

⚠️ Incluso un modelo regularizado (como Ridge o Lasso) necesita una correcta división para evitar falsas expectativas de desempeño.

### ✅ **Reflexión Final**

> Ridge y Lasso no solo ajustan una línea, sino que le dicen al modelo: **“No te sobrecompliques”**.
>  Compararlos te ayuda a elegir el balance ideal entre **simplicidad y precisión**.
>  ¡Y recuerda siempre dividir tus datos antes de entrenar! ✂️📚

### 💡 Tips:

- 📌 Usa `GridSearchCV` para encontrar el mejor valor de `alpha`.
- 🔄 Repite con validación cruzada para mayor robustez.
- 📎 Observa qué coeficientes anula Lasso: es útil para **selección de características**.