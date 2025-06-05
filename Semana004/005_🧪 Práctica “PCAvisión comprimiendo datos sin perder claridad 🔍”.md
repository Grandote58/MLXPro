# 🧪 **Práctica: “PCAvisión: comprimiendo datos sin perder claridad 🔍”**

### 🎯 **Objetivo de aprendizaje**

Aplicar el algoritmo **PCA (Análisis de Componentes Principales)** sobre un conjunto de datos real para reducir la dimensionalidad de forma efectiva y visualizar agrupaciones ocultas en un espacio 2D.

### 📂 **Dataset real: Iris Dataset**

- 🏷️ Nombre: *Iris Species Dataset*
- 📦 Fuente: Kaggle
- 📥 URL: https://www.kaggle.com/datasets/uciml/iris
- Archivo: `Iris.csv`

💡 Este conjunto de datos contiene **150 muestras** de flores con 4 variables: *sepal length, sepal width, petal length, petal width*, y la especie como etiqueta.

## 👨‍🏫 **PASO A PASO EN GOOGLE COLAB**

### 🔹 1. Importar librerías

```python
# 📚 Importamos las librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
```

### 🔹 2. Cargar el dataset

🔽 **Sube el archivo `Iris.csv` manualmente a Google Colab** desde el panel lateral.

```python
# 📥 Cargar el dataset de Iris
df = pd.read_csv("Iris.csv")

# 👁️ Visualizar los primeros registros
df.head()
```

### 🔹 3. Explorar los datos

```python
# 🔍 Información general del dataset
df.info()

# 🧮 Estadísticas descriptivas
df.describe()

# 🎨 Visualización por especie
sns.pairplot(df, hue='Species')
plt.suptitle("Distribución original por especie 🌸", y=1.02)
plt.show()
```

### 🔹 4. Preparar los datos para PCA

```python
# ✂️ Eliminamos columnas no necesarias
df_features = df.drop(columns=['Id', 'Species'])

# 🧽 Estandarizamos los datos
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_features)

# 📌 Revisamos cómo lucen los datos escalados
pd.DataFrame(df_scaled, columns=df_features.columns).head()
```

### 🔹 5. Aplicar PCA

```python
# 🧠 Aplicamos PCA con 2 componentes
pca = PCA(n_components=2)
pca_components = pca.fit_transform(df_scaled)

# 🧾 Convertimos a DataFrame para visualizar
df_pca = pd.DataFrame(data=pca_components, columns=['PC1', 'PC2'])
df_pca['Species'] = df['Species']
df_pca.head()
```

### 🔹 6. Varianza explicada por componente

```python
# 📊 Gráfico de varianza explicada
explained_variance = pca.explained_variance_ratio_

plt.figure(figsize=(6,4))
plt.bar(['PC1', 'PC2'], explained_variance * 100, color='skyblue')
plt.title("Varianza explicada por componente 🎯")
plt.ylabel('% de Varianza')
plt.show()

# 🔢 Porcentaje acumulado
print(f"Total varianza explicada: {explained_variance.sum() * 100:.2f}%")
```

### 🔹 7. Visualización 2D del resultado de PCA

```python
# 🎨 Graficar los datos proyectados en los primeros dos componentes
plt.figure(figsize=(8,6))
sns.scatterplot(x='PC1', y='PC2', hue='Species', data=df_pca, palette='Set2')
plt.title("🌼 Datos reducidos con PCA (PC1 vs PC2)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.grid(True)
plt.legend(title='Especie')
plt.show()
```

## 🧠 Reflexión 

- ¿Qué tan bien logra PCA separar las especies de flores en 2 dimensiones?
- ¿Qué ocurre con las especies que se traslapan?
- ¿Por qué es importante estandarizar los datos antes de aplicar PCA?

## 📎 Conclusión

Con esta práctica lograste:

 ✅ Escalar los datos

 ✅ Aplicar reducción de dimensionalidad

 ✅ Visualizar resultados de manera clara y compacta

 ✅ Comprender cómo PCA conserva patrones esenciales

🧠 *“Menos dimensiones, más comprensión.”*