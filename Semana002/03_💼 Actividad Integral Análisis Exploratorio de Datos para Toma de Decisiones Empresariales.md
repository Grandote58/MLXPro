# 💼 **Actividad Integral: Análisis Exploratorio de Datos para Toma de Decisiones Empresariales**

### 🧠 Contexto Empresarial

📍 **Solicitud del área de negocios**:
 *El equipo de marketing de una cadena de supermercados quiere entender mejor el perfil de sus clientes para mejorar campañas personalizadas y optimizar el programa de fidelización.*

## 📁 Paso 1: **Identificación del Repositorio de Datos**

Utilizaremos el dataset **Customer Personality Analysis** del repositorio de datos abiertos de Kaggle, que contiene información demográfica, financiera y de comportamiento de clientes.

🔗 **Dataset directo CSV**:
 https://raw.githubusercontent.com/joaquinamatrodriguez/Customer-Analysis/main/marketing_campaign.csv

## 🚀 **Objetivo del Análisis Exploratorio (EDA)**

- Limpiar y preparar los datos.
- Analizar estadísticas descriptivas (media, mediana, moda, varianza, asimetría, curtosis).
- Visualizar patrones y correlaciones.
- Detectar y manejar valores extremos.
- Aplicar transformaciones donde sea necesario.

## 🛠️ **Guía Paso a Paso (Google Colab)**

### 🔹 **1. Cargar librerías y datos**

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

url = 'https://raw.githubusercontent.com/joaquinamatrodriguez/Customer-Analysis/main/marketing_campaign.csv'
df = pd.read_csv(url, sep=';')
df.head()
```

### 🔹 **2. Limpieza de datos básica** 🧹

```python
# Convertir fechas
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])

# Eliminar columnas irrelevantes
df.drop(columns=['ID', 'Z_CostContact', 'Z_Revenue'], inplace=True)

# Verificar valores faltantes
print(df.isnull().sum())
df = df.dropna()  # eliminar nulos para simplificación en esta práctica
```

### 🔹 **3. Análisis individual de variables (ejemplo: Income)** 📊

```python
var = 'Income'
print(f"Media: {df[var].mean():.2f}")
print(f"Mediana: {df[var].median():.2f}")
print(f"Moda: {df[var].mode().values[0]:.2f}")
print(f"Varianza: {df[var].var():.2f}")
print(f"Asimetría: {skew(df[var]):.2f}")
print(f"Curtosis: {kurtosis(df[var]):.2f}")
```

### 🔹 **4. Visualización de patrones** 🖼️

#### a. Histograma y KDE

```python
sns.histplot(df[var], kde=True)
plt.title("Distribución de Ingresos")
plt.show()
```

#### b. Boxplot para outliers

```python
sns.boxplot(x=df[var])
plt.title("Boxplot de Ingresos")
plt.show()
```

### 🔹 **5. Análisis grupal entre variables** 🔗

```python
# Relación entre ingreso y gastos en vino
sns.scatterplot(data=df, x='Income', y='MntWines')
plt.title("Gasto en Vino vs Ingreso")
plt.show()

# Mapa de calor
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Matriz de Correlación")
plt.show()
```

### 🔹 **6. Transformación de variables sesgadas** 🔄

```python
# Ver antes
print("Asimetría original:", skew(df['Income']))

# Aplicar logaritmo
df['Income_log'] = np.log(df['Income'])

# Ver después
print("Asimetría corregida:", skew(df['Income_log']))

# Visualizar
sns.histplot(df['Income_log'], kde=True)
plt.title("Distribución Ingreso Log-transformado")
plt.show()
```

## 🎓 Reflexión y preguntas para reforzar el aprendizaje

1. ¿Qué variables tienen mayor dispersión y por qué?
2. ¿Qué relaciones importantes observaste entre ingresos y productos?
3. ¿Qué beneficio aporta aplicar transformaciones logarítmicas?
4. ¿Cómo te ayuda este EDA para construir un modelo predictivo?