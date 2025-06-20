# **“¿Regresamos o Clasificamos? 🧠📏 Evaluando Modelos con Métricas Inteligentes”**

### 🎯 **Objetivo**

Entrenar dos modelos sobre el dataset **California Housing**:

1. Un modelo de **regresión** para predecir el valor promedio de una vivienda.
2. Un modelo de **clasificación binaria**, transformando el target en:
   - 🟢 "1" si el precio ≥ 2.0
   - 🔴 "0" si el precio < 2.0

Se evaluarán ambos con sus métricas específicas y se compararán los resultados visualmente.

### 📦 **Dataset**

Se utiliza el dataset oficial de `sklearn.datasets`.

🔗 URL oficial:
 https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

### 🧰 **Paso a Paso**

#### 🔹 Paso 1: Importar librerías necesarias

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
```

#### 🔹 Paso 2: Cargar y preparar los datos

```python
# Cargar dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# Escalar datos
X = df.drop(columns='MedHouseVal')
y_reg = df['MedHouseVal']  # para regresión

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Crear variable binaria para clasificación
y_clf = (y_reg >= 2.0).astype(int)
```

### 🧮 MODELO 1: REGRESIÓN

#### 🔹 Paso 3: División y entrenamiento regresión

```python
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)
model_reg = LinearRegression()
model_reg.fit(X_train_r, y_train_r)
y_pred_r = model_reg.predict(X_test_r)
```

#### 🔹 Paso 4: Evaluación regresión

```python
mse = mean_squared_error(y_test_r, y_pred_r)
mae = mean_absolute_error(y_test_r, y_pred_r)
r2 = r2_score(y_test_r, y_pred_r)

print(f"📉 MSE: {mse:.4f}")
print(f"📊 MAE: {mae:.4f}")
print(f"📈 R²: {r2:.4f}")
```

#### 🔹 Paso 5: Visualización regresión

```python
plt.figure(figsize=(7,5))
sns.scatterplot(x=y_test_r, y=y_pred_r, alpha=0.5)
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--')
plt.xlabel("Valor Real 🏠")
plt.ylabel("Valor Predicho 💸")
plt.title("Regresión: Real vs Predicho 📈")
plt.grid(True)
plt.show()
```

### 🔢 MODELO 2: CLASIFICACIÓN

#### 🔹 Paso 6: División y entrenamiento clasificación

```python
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_scaled, y_clf, test_size=0.2, random_state=42)
model_clf = LogisticRegression()
model_clf.fit(X_train_c, y_train_c)
y_pred_c = model_clf.predict(X_test_c)
```

#### 🔹 Paso 7: Evaluación clasificación

```python
acc = accuracy_score(y_test_c, y_pred_c)
prec = precision_score(y_test_c, y_pred_c)
rec = recall_score(y_test_c, y_pred_c)
f1 = f1_score(y_test_c, y_pred_c)

print(f"✅ Accuracy: {acc:.4f}")
print(f"🔍 Precision: {prec:.4f}")
print(f"📢 Recall: {rec:.4f}")
print(f"🎯 F1-score: {f1:.4f}")
```

#### 🔹 Paso 8: Matriz de Confusión

```python
cm = confusion_matrix(y_test_c, y_pred_c)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["< 2.0", "≥ 2.0"])
disp.plot(cmap="Blues")
plt.title("🔵 Matriz de Confusión - Clasificación")
plt.show()
```

### ✅ **Reflexión final**

> **No existe una métrica universal**. Cada métrica responde una pregunta diferente.
>  **Evalúa con varias métricas**, visualiza errores y elige el modelo que mejor se ajuste a tu problema real.
>  ¡Tus decisiones de negocio o ciencia deben basarse en datos... y en buenas métricas! 🔬📉📊