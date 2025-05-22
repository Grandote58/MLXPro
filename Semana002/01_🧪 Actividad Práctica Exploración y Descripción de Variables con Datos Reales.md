# 🧪 **Actividad Práctica: Exploración y Descripción de Variables con Datos Reales**

🎯 **Objetivo**: Aprender a describir y visualizar variables reales utilizando técnicas estadísticas y gráficas fundamentales para machine learning.

📦 **Dataset**: Datos de precios de viviendas
 🔗 Fuente: OpenML Housing dataset (URL directa CSV)

### 🛠️ **Pasos detallados en Google Colab**

### 🔹 **Paso 1: Cargar librerías necesarias**

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
```

------

### 🔹 **Paso 2: Cargar datos desde una URL** 🌐

```python
url = "https://www.openml.org/data/get_csv/31/dataset_31_car.arff"
df = pd.read_csv(url)
df.head()
```

📌 Verifica las primeras filas del dataset para conocer sus variables.

### 🔹 **Paso 3: Descripción estadística básica** 📊

```python
df.describe()
```

✅ Observa: media, mediana, desviación estándar, mínimo, máximo.

### 🔹 **Paso 4: Análisis de dispersión y distribución** 📉

```python
# Ejemplo con la variable 'doors'
variable = 'doors'
print("Asimetría:", df[variable].skew())
print("Curtosis:", df[variable].kurt())

# Histograma
sns.histplot(df[variable], kde=True)
plt.title(f'Distribución de {variable}')
plt.show()
```

🧠 Interpreta si los datos están sesgados o tienen colas pesadas.

### 🔹 **Paso 5: Visualizar outliers con Boxplot** 🎯

```python
sns.boxplot(x=df[variable])
plt.title(f'Boxplot de {variable}')
plt.show()
```

🔍 Observa los puntos fuera de los bigotes: esos son **outliers**.

### 🔹 **Paso 6: Relaciones entre variables (scatter plot)** 🔗

```python
sns.scatterplot(data=df, x='persons', y='lug_boot')
plt.title('Relación entre persons y lug_boot')
plt.show()
```

📈 Identifica si hay correlaciones positivas o negativas.

### 🔹 **Paso 7: Mapa de calor de correlación** 🔥

```python
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Mapa de Correlación entre Variables")
plt.show()
```

🌡️ Muy útil para seleccionar variables importantes en un modelo.

### 🎓 **Reflexión final**

- ¿Qué variable presenta mayor dispersión?
- ¿Hay evidencia de sesgo o curtosis elevada?
- ¿Qué transformaciones aplicarías antes de modelar?

