# **Predicción de Costos Médicos 💸🩺: Entrenamiento de Modelos de Machine Learning en Salud Pública”**

### 🎯 **Objetivo General**

Construir un modelo predictivo para estimar los **costos médicos** de pacientes, utilizando variables como edad, IMC, tabaquismo y región. Se aplicarán técnicas modernas de entrenamiento, validación cruzada, escalado y regularización para optimizar el modelo y evaluar su desempeño.

### 📂 **Dataset utilizado**

📊 Dataset: **Medical Cost Personal Dataset**
 📌 Fuente: Kaggle (100% Open Access)

📥 Descarga directa del CSV (solo si se sube desde Google Colab):

```python
!wget https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv
```

### 📘 **Paso a paso con documentación**

#### 🔹 Paso 1: Importar librerías necesarias 🧰

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```

#### 🔹 Paso 2: Cargar y explorar los datos 🩺

```python
# Cargar el dataset
df = pd.read_csv('insurance.csv')
df.head()
pythonCopiarEditar# Ver estructura y valores únicos
df.info()
df.describe()
df['region'].value_counts()
```

#### 🔹 Paso 3: Preprocesamiento (escalado y codificación) ⚖️

```python
# Separar X e y
X = df.drop(columns='charges')
y = df['charges']

# Columnas numéricas y categóricas
num_cols = ['age', 'bmi', 'children']
cat_cols = ['sex', 'smoker', 'region']

# Pipeline de transformación
preprocessor = ColumnTransformer([
    ('scaler', StandardScaler(), num_cols),
    ('encoder', OneHotEncoder(drop='first'), cat_cols)
])
```

#### 🔹 Paso 4: División en training y test ✂️

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### 🔹 Paso 5: Entrenar modelos con regularización 🧠

```python
# Pipeline completo con Ridge
ridge_pipe = Pipeline([
    ('preprocess', preprocessor),
    ('model', Ridge(alpha=1.0))
])

# Pipeline con Lasso
lasso_pipe = Pipeline([
    ('preprocess', preprocessor),
    ('model', Lasso(alpha=0.1))
])
```

#### 🔹 Paso 6: Validación cruzada 🔁

```python
# Evaluación con cross-validation (5 folds)
cv_ridge = cross_val_score(ridge_pipe, X_train, y_train, cv=5, scoring='r2')
cv_lasso = cross_val_score(lasso_pipe, X_train, y_train, cv=5, scoring='r2')

print(f"🔁 Ridge R² promedio: {cv_ridge.mean():.4f}")
print(f"🔁 Lasso R² promedio: {cv_lasso.mean():.4f}")
```

#### 🔹 Paso 7: Evaluación final del modelo 📈

```python
# Entrenar modelos
ridge_pipe.fit(X_train, y_train)
lasso_pipe.fit(X_train, y_train)

# Predicciones
y_pred_ridge = ridge_pipe.predict(X_test)
y_pred_lasso = lasso_pipe.predict(X_test)

# Métricas
def evaluar(y_true, y_pred, modelo):
    print(f"📊 Evaluación para {modelo}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"MSE: {mean_squared_error(y_true, y_pred):.2f}")
    print(f"R²: {r2_score(y_true, y_pred):.4f}\n")

evaluar(y_test, y_pred_ridge, "Ridge")
evaluar(y_test, y_pred_lasso, "Lasso")
```

#### 🔹 Paso 8: Visualización de resultados 📊

```python
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_ridge, label='Ridge', alpha=0.6)
plt.scatter(y_test, y_pred_lasso, label='Lasso', alpha=0.6)
plt.plot([0, 50000], [0, 50000], 'r--')
plt.xlabel("Costo médico real 💸")
plt.ylabel("Costo predicho 🧮")
plt.title("Comparación de predicción: Ridge vs Lasso")
plt.legend()
plt.grid(True)
plt.show()
```

### 🎤 **Tips para practicar más** 👨‍🏫✨

✅ Prueba otros valores de `alpha` en Ridge y Lasso usando `GridSearchCV`.

✅ Añade nuevas variables artificiales (como IMC x edad).

✅ Intenta convertir `charges` en una variable categórica (ej. "alto", "bajo") y aplica clasificación.

✅ Implementa la misma práctica con modelos no lineales (árboles, SVR).

✅ Crea una función para comparar varias métricas automáticamente.

### ✅ **Reflexión final**

> Esta práctica muestra cómo combinar los elementos esenciales del ciclo de Machine Learning para un problema sanitario real: predicción de costos médicos.
>  Cada paso (escalado, regularización, validación, evaluación) contribuye a crear un modelo más **robusto, justo y replicable** en escenarios del mundo real de salud pública 🩺📊