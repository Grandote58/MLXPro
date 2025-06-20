# **¡Todo en la misma escala! 📐📊 Comparando Normalización y Estandarización en Datos de Viviendas 🏡**

### 🎯 **Objetivo**

Aplicar **Normalización** y **Estandarización** sobre dos variables del dataset **California Housing**, visualizar los cambios y entrenar un modelo de regresión lineal para predecir el precio medio de una vivienda. Se evidenciará cómo el escalado mejora el rendimiento y estabilidad del modelo.

### 📦 **Dataset**

Se utilizará el dataset **California Housing** de `sklearn.datasets`.

🔗 URL oficial:
 https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

### 🧪 **Paso a Paso**

#### 🔹 Paso 1: Importar librerías necesarias 🧰

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
```

#### 🔹 Paso 2: Cargar el dataset 🏡

```python
# Cargar datos
data = fetch_california_housing(as_frame=True)
df = data.frame

# Visualizar las primeras filas
df.head()
```

#### 🔹 Paso 3: Seleccionar y visualizar las variables originales 📊

```python
# Seleccionar dos variables: AveRooms y AveOccup
features = df[['AveRooms', 'AveOccup']]

# Graficar distribución original
features.hist(figsize=(10, 4), bins=30, color='lightblue')
plt.suptitle("🔍 Distribución original de AveRooms y AveOccup")
plt.show()
```

#### 🔹 Paso 4: Aplicar Normalización y Estandarización ⚖️

```python
# Normalización
minmax_scaler = MinMaxScaler()
features_norm = pd.DataFrame(minmax_scaler.fit_transform(features), columns=features.columns)

# Estandarización
standard_scaler = StandardScaler()
features_std = pd.DataFrame(standard_scaler.fit_transform(features), columns=features.columns)
```

#### 🔹 Paso 5: Visualizar escalado con gráficas 🎨

```python
# Graficar resultados de escalado
plt.figure(figsize=(12, 4))

# Normalización
plt.subplot(1, 2, 1)
plt.hist(features_norm, bins=30, stacked=True)
plt.title("🔵 Normalización (MinMaxScaler)")

# Estandarización
plt.subplot(1, 2, 2)
plt.hist(features_std, bins=30, stacked=True)
plt.title("🟠 Estandarización (StandardScaler)")

plt.tight_layout()
plt.show()
```

#### 🔹 Paso 6: Entrenar modelo sin escalado vs con escalado 📈

```python
# Variable objetivo
y = df['MedHouseVal']

# Modelos con variables originales
X_train_o, X_test_o, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)
model_o = LinearRegression().fit(X_train_o, y_train)
pred_o = model_o.predict(X_test_o)
r2_o = r2_score(y_test, pred_o)

# Modelos con normalización
X_train_n, X_test_n, _, _ = train_test_split(features_norm, y, test_size=0.2, random_state=42)
model_n = LinearRegression().fit(X_train_n, y_train)
pred_n = model_n.predict(X_test_n)
r2_n = r2_score(y_test, pred_n)

# Modelos con estandarización
X_train_s, X_test_s, _, _ = train_test_split(features_std, y, test_size=0.2, random_state=42)
model_s = LinearRegression().fit(X_train_s, y_train)
pred_s = model_s.predict(X_test_s)
r2_s = r2_score(y_test, pred_s)

# Resultados
print(f"📊 Sin escalado - R²: {r2_o:.4f}")
print(f"🔵 Normalizado - R²: {r2_n:.4f}")
print(f"🟠 Estandarizado - R²: {r2_s:.4f}")
```

### 💬 **Explicación**

🔍 El escalado es como dar las mismas oportunidades a todas las variables. Si una tiene un rango de 0–1000 y otra de 0–10, el modelo le dará más peso a la primera... aunque no sea más importante.

🧪 En esta práctica:

- Usamos dos variables (`AveRooms` y `AveOccup`) para simplificar la visualización.
- Aplicamos dos técnicas:
  - **MinMaxScaler (Normalización)**: ajusta los valores entre 0 y 1.
  - **StandardScaler (Estandarización)**: centra la media en 0 y ajusta la desviación a 1.

📊 Luego de aplicar cada técnica:

- **Visualizamos sus efectos** con histogramas.
- **Entrenamos modelos** con los datos escalados y comparamos su desempeño con el modelo sin escalado.

📈 Como se observa en los resultados, **el escalado puede mejorar el rendimiento o la estabilidad**, especialmente en modelos que utilizan operaciones matemáticas con distancias o pesos.

### ✅ **Reflexión final**

> El preprocesamiento es más que una etapa técnica: es **la base para que los modelos aprendan correctamente**.
>  Escalar las variables es como nivelar el terreno antes de construir: garantiza un entrenamiento más justo, más rápido y más preciso. 🧱⚖️📐

### 💡 Tips :

- Usa `MinMaxScaler` si tus datos no tienen valores extremos (outliers).
- Usa `StandardScaler` si esperas una distribución normal o tienes outliers.
- Aplica escalado **después** de la división Train/Test para evitar fuga de datos.