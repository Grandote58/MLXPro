# 🧪 **Actividad Práctica : Análisis Exploratorio de Datos de Salud Pública**

🎯 **Objetivo**: Aplicar técnicas de descripción de variables para interpretar y preparar datos reales del mundo de la salud antes de modelarlos.

📘 **Caso real**: Predecir si una persona tiene diabetes a partir de variables como glucosa, presión sanguínea y masa corporal.

🔗 **Dataset**: Pima Indians Diabetes Database
 📂 **Fuente directa CSV**: https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv

### 🔹 **PASO 1: Cargar librerías necesarias**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
```

### 🔹 **PASO 2: Cargar y explorar el dataset desde la URL**

```python
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)
df.head()
```

🧾 Verifica los nombres de las columnas: `'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'`.

### 🔹 **PASO 3: Estadísticas Descriptivas Globales** 📊

```python
df.describe()
```

💡 Revisa:

- Media, mediana, desviación estándar.
- Identifica columnas con rangos extremos o ceros sospechosos.

### 🔹 **PASO 4: Análisis de una variable numérica paso a paso (ejemplo: Glucose)**

#### a. 📌 **Resumen estadístico individual**

```python
var = 'Glucose'
print(f"Asimetría de {var}:", df[var].skew())
print(f"Curtosis de {var}:", df[var].kurt())
```

#### b. 📈 **Histograma + KDE**

```python
sns.histplot(df[var], kde=True, bins=30, color="skyblue")
plt.title(f'Distribución de {var}')
plt.show()
```

#### c. 🎯 **Boxplot para detectar outliers**

```python
sns.boxplot(x=df[var], color="lightcoral")
plt.title(f'Boxplot de {var}')
plt.show()
```

### 🔹 **PASO 5: Identificar outliers con reglas estadísticas**

```python
Q1 = df[var].quantile(0.25)
Q3 = df[var].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df[var] < Q1 - 1.5 * IQR) | (df[var] > Q3 + 1.5 * IQR)]
print(f"Número de outliers en {var}:", len(outliers))
```

### 🔹 **PASO 6: Comparar dos variables con scatter plot** 🔗

```python
sns.scatterplot(data=df, x="BMI", y="Glucose", hue="Outcome", palette="coolwarm")
plt.title("Relación entre BMI y Glucosa por diagnóstico")
plt.show()
```

👀 Observa si hay patrones de clasificación visuales.

### 🔹 **PASO 7: Matriz de correlación (Heatmap)**

```python
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Mapa de Calor de Correlaciones")
plt.show()
```

### 📘 **Actividad de reflexión**

1. ¿Qué variables tienen mayor asimetría o curtosis?
2. ¿Cómo influye el rango de valores en la preparación de datos?
3. ¿Qué variables podrían requerir transformación antes de usarse en un modelo?