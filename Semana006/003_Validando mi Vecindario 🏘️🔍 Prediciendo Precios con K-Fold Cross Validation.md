# **Validando mi Vecindario 🏘️🔍: Prediciendo Precios con K-Fold Cross Validation**

### 🎯 **Objetivo**

Aprender a implementar **K-Fold Cross Validation** para evaluar el desempeño de un modelo de regresión lineal sobre el dataset **California Housing**, observando los resultados en cada fold para analizar su estabilidad y capacidad de generalización.

### 📦 **Dataset**

Se usará el dataset real de California Housing.

🔗 URL oficial:
 https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

### 🧪 **Paso a paso en Google Colab**

#### 🔹 Paso 1: Importar librerías 🧰

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
```

#### 🔹 Paso 2: Cargar el dataset 🏡

```python
# Cargar datos de California Housing
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# Mostrar estructura del dataset
X.head()
```

#### 🔹 Paso 3: Escalar las características ⚖️

```python
# Escalar los datos para mejorar el rendimiento del modelo
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

#### 🔹 Paso 4: Configurar K-Fold y modelo 🤖

```python
# Crear modelo de regresión
modelo = LinearRegression()

# Configurar validación cruzada con K=5
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
```

#### 🔹 Paso 5: Ejecutar Validación Cruzada y evaluar 🔄📏

```python
# Validar usando R² como métrica
scores = cross_val_score(modelo, X_scaled, y, cv=kfold, scoring='r2')

# Mostrar resultados por fold
for i, score in enumerate(scores, 1):
    print(f"📂 Fold {i} - R²: {score:.4f}")

# Promedio y desviación estándar
print(f"\n✅ Promedio de R²: {scores.mean():.4f}")
print(f"📉 Desviación estándar de R²: {scores.std():.4f}")
```

#### 🔹 Paso 6: Visualizar los resultados por Fold 📊

```python
# Visualización de los puntajes
plt.figure(figsize=(8, 5))
plt.bar(range(1, 6), scores, color='skyblue')
plt.axhline(scores.mean(), color='red', linestyle='--', label='Promedio R²')
plt.title("📊 Resultados de R² por Fold (K-Fold CV)")
plt.xlabel("Fold")
plt.ylabel("R² Score")
plt.ylim(0.4, 0.8)
plt.legend()
plt.show()
```

### 💬 **Explicación : ¿Por qué usar K-Fold Cross Validation?**

🔁 Cuando dividimos los datos una sola vez en train/test, nuestro resultado puede **variar según la suerte**. 

¿Qué pasa si justo los datos difíciles quedaron en el set de prueba? 🤷‍♂️

💡 **K-Fold Cross Validation** divide el dataset en **K partes**, entrena y evalúa **K veces**, cada vez con un grupo distinto de prueba.

✔️ Así obtenemos **K evaluaciones independientes**, y al calcular el **promedio**, tenemos una medida más **estable y confiable** del rendimiento del modelo.

📌 En esta práctica usamos **R²**, que indica qué tan bien el modelo explica la variabilidad del precio de la vivienda. Un R² alto y estable entre folds es buena señal 🧠📈

### ✅ **Reflexión final**

> Validar con K-Fold es como hacer un chequeo médico en cinco clínicas diferentes: si todos coinciden, el diagnóstico es confiable 🏥✅
>  Esta técnica no solo mejora la evaluación, sino que es base para métodos avanzados como optimización de hiperparámetros y ensamblado de modelos.
>  ¡Nunca más confíes en un solo split de datos! 😉

### 💡 Tips :

- Usa `shuffle=True` para que los folds sean aleatorios.
- Combínalo con `GridSearchCV` para elegir los mejores parámetros.
- Aumenta `n_splits` si tienes muchos datos; usa `K=10` para validación más fina.