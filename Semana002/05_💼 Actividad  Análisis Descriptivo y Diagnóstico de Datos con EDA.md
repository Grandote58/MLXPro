# 💼 **Actividad : Análisis Descriptivo y Diagnóstico de Datos con EDA**

### 🧠 **Contexto Formativo y Empresarial**

📍 *El equipo de Business Intelligence de una aseguradora desea comprender a fondo el perfil de sus asegurados para optimizar campañas de prevención y ajustar tarifas.*

## 🗃️ **Dataset real para la práctica**

📘 **Nombre:** Insurance Dataset
 🔗 **URL directa:**
 https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv

## 🎯 **Objetivos de aprendizaje**

- Clasificar variables (categóricas vs continuas).
- Calcular medidas de tendencia central y dispersión.
- Realizar EDA detallado con apoyo visual.
- Tratar valores nulos y errores.
- Detectar patrones invisibles con gráficos.
- Preparar datos de forma profesional para modelado.

## 🛠️ **Guía Detallada en Google Colab / Jupyter Notebook**

### 🔹 **Paso 1: Cargar librerías y datos**

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
pythonCopiarEditarurl = 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv'
df = pd.read_csv(url)
df.head()
```

### 🔹 **Paso 2: Clasificación de variables** 🔍

```python
df.info()
```

📘 **Conceptos**:

- **Variable categórica**: representa categorías (sin orden numérico).
- **Variable continua**: valores numéricos con rango amplio y escalable.

📊 **Clasificación rápida**:

| Variable | Tipo       | Descripción              |
| -------- | ---------- | ------------------------ |
| age      | Continua   | Edad del asegurado       |
| sex      | Categórica | Género                   |
| bmi      | Continua   | Índice de masa corporal  |
| children | Discreta   | Nº de hijos dependientes |
| smoker   | Categórica | Si fuma o no             |
| region   | Categórica | Región geográfica        |
| charges  | Continua   | Costo del seguro         |

### 🔹 **Paso 3: Medidas estadísticas descriptivas**

```python
df.describe()
```

📘 **Conceptos clave**:

- **Media (mean)**: promedio de los datos.
- **Mediana**: valor central.
- **Moda**: valor más frecuente.
- **Varianza**: dispersión respecto a la media.
- **Desviación estándar**: dispersión general.
- **Asimetría**: indica sesgo hacia derecha o izquierda.
- **Curtosis**: mide qué tan “puntiaguda” es la distribución.

```python
for var in ['age', 'bmi', 'charges']:
    print(f"\nEstadísticas de {var}")
    print(f"Media: {df[var].mean():.2f}")
    print(f"Mediana: {df[var].median():.2f}")
    print(f"Moda: {df[var].mode()[0]:.2f}")
    print(f"Varianza: {df[var].var():.2f}")
    print(f"Asimetría: {skew(df[var]):.2f}")
    print(f"Curtosis: {kurtosis(df[var]):.2f}")
```

### 🔹 **Paso 4: Tratamiento de valores faltantes (Missing values)** 🧹

```python
df.isnull().sum()
```

📘 **Concepto nuevo**:

- **Missing values**: datos ausentes en el dataset.
   Se pueden imputar (rellenar) o eliminar según su proporción e impacto.

💡 En este dataset no hay missing, pero si hubiera, podríamos hacer:

```python
df['bmi'].fillna(df['bmi'].median(), inplace=True)
```

### 🔹 **Paso 5: Visualización para detectar patrones invisibles**

#### a. Histograma + KDE

```python
sns.histplot(df['charges'], kde=True)
plt.title('Distribución de cargos médicos')
plt.show()
```

#### b. Boxplot para outliers

```python
sns.boxplot(x=df['charges'])
plt.title('Boxplot de cargos médicos')
plt.show()
```

#### c. Diagrama de dispersión entre variables numéricas

```python
sns.scatterplot(data=df, x='bmi', y='charges', hue='smoker')
plt.title("Relación entre BMI y Costo del seguro (color fumador)")
plt.show()
```

#### d. Relación entre variables categóricas y numéricas

```python
sns.boxplot(x='smoker', y='charges', data=df)
plt.title("Costo del seguro según hábito de fumar")
plt.show()
```

### 🔹 **Paso 6: Identificación y eliminación de errores**

📘 **Errores comunes en datos reales**:

- Edades fuera de rango lógico.
- BMI menor a 10 o mayor a 60.
- Cargos ($charges$) negativos o excesivos sin justificación.

```python
df = df[(df['age'] > 0) & (df['bmi'] > 10) & (df['charges'] > 0)]
```

### 🔹 **Paso 7: Transformaciones si hay violación de supuestos**

#### Ver sesgo antes y después

```python
print("Asimetría original:", skew(df['charges']))

df['charges_log'] = np.log(df['charges'])

print("Asimetría corregida:", skew(df['charges_log']))

sns.histplot(df['charges_log'], kde=True)
plt.title("Distribución de cargos transformada (log)")
plt.show()
```

## 🎓 Reflexión didáctica final

1. ¿Cómo identificaste variables importantes para el análisis?
2. ¿Qué patrones encontraste al cruzar variables?
3. ¿Qué transformación ayudó a normalizar tus datos?
4. ¿Cómo te ayudan estas técnicas para preparar un dataset de modelado?