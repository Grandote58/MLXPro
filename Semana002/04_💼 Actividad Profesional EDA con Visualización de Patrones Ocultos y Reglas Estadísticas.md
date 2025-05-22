# 💼 **Actividad : EDA con Visualización de Patrones Ocultos y Reglas Estadísticas**

### 🧠 **Contexto Empresarial**

📍 **Caso empresarial**:
 *El departamento de riesgos financieros desea analizar los patrones de comportamiento de préstamos personales para ajustar sus políticas de crédito y prevenir incumplimientos.*

## 📁 **Datos abiertos seleccionados**

📊 **Dataset**: Customer Bank Loan
 🔗 **URL directa CSV**:
 https://raw.githubusercontent.com/selva86/datasets/master/Bank_Personal_Loan_Modelling.csv

## 🎯 **Objetivo**

- Realizar un análisis descriptivo profesional y visual.
- Detectar patrones complejos e invisibles a simple vista.
- Aplicar reglas estadísticas para aislar valores extremos.
- Preparar variables para futuros modelos predictivos.

## ✅ **Guía detallada paso a paso – Google Colab**

### 🔹 **1. Cargar librerías y datos**

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

url = 'https://raw.githubusercontent.com/selva86/datasets/master/Bank_Personal_Loan_Modelling.csv'
df = pd.read_csv(url)
df.drop(columns=['ID', 'ZIP Code'], inplace=True)
df.head()
```

### 🔹 **2. Estadística descriptiva detallada**

```python
var = 'Income'
print(f"Media: {df[var].mean():.2f}")
print(f"Mediana: {df[var].median():.2f}")
print(f"Moda: {df[var].mode()[0]:.2f}")
print(f"Varianza: {df[var].var():.2f}")
print(f"Asimetría: {skew(df[var]):.2f}")
print(f"Curtosis: {kurtosis(df[var]):.2f}")
```

### 🔹 **3. Detección de outliers con reglas estadísticas** 🧠

```python
Q1 = df[var].quantile(0.25)
Q3 = df[var].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df[var] < Q1 - 1.5*IQR) | (df[var] > Q3 + 1.5*IQR)]
print(f"Outliers encontrados en {var}: {len(outliers)}")
```

### 🔹 **4. Visualizaciones para detectar patrones invisibles**

#### a. 📈 **Histograma + KDE (densidad)**

```python
sns.histplot(df[var], kde=True, color='skyblue')
plt.title(f'Distribución y densidad de {var}')
plt.show()
```

#### b. 📦 **Boxplot de múltiples variables**

```python
features = ['Income', 'CCAvg', 'Mortgage']
for f in features:
    sns.boxplot(x=df[f])
    plt.title(f'Boxplot de {f}')
    plt.show()
```

#### c. 📊 **Gráfico de dispersión con color por Loan**

```python
sns.scatterplot(data=df, x='Income', y='CCAvg', hue='Personal Loan')
plt.title("Relación entre Income y CCAvg por tipo de préstamo")
plt.show()
```

### 🔹 **5. Heatmap de correlación para relaciones no evidentes** 🔥

```python
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Mapa de calor de correlación")
plt.show()
```

### 🔹 **6. Transformaciones para mejorar distribución**

```python
df['Income_log'] = np.log(df['Income'] + 1)

sns.histplot(df['Income_log'], kde=True)
plt.title("Distribución de Ingresos Transformada (Log)")
plt.show()

print("Asimetría original:", skew(df['Income']))
print("Asimetría corregida:", skew(df['Income_log']))
```

## 🎓 **Preguntas para práctica y reflexión**

1. ¿Qué variables presentan distribución sesgada o curtosis anormal?
2. ¿Cómo ayuda la visualización a detectar patrones que una tabla no muestra?
3. ¿Qué relaciones detectaste entre ingresos y otros factores como préstamos?
4. ¿Qué tipo de transformaciones mejoraron la simetría de las variables?