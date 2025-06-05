# 🧪 **Práctica: “PCA Sommelier 🍷: Catando Vinos con Análisis de Componentes Principales”**

### 🎯 **Objetivo**

Aplicar el algoritmo **PCA (Principal Component Analysis)** para reducir la dimensionalidad de un conjunto de datos de características químicas de vinos y visualizar la separación entre distintas variedades.

## 📂 **Dataset real: Wine Dataset**

- **📥 Fuente:** Kaggle – Wine Dataset (UCI original)
- **🔗 URL descarga directa:**
   👉 https://www.kaggle.com/datasets/rajyellow46/wine-quality
- Archivos:
  - `winequality-red.csv` (1599 muestras)
  - `winequality-white.csv` (4898 muestras)

📌 En esta práctica usaremos el archivo `winequality-red.csv` para simplificar el análisis inicial.

## 👨‍🏫 **Paso a Paso en Google Colab**

### 🔹 1. Importar librerías necesarias

```python
# 📚 Librerías de análisis y visualización
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
```

### 🔹 2. Cargar el dataset

🔽 **Sube manualmente el archivo `winequality-red.csv` a Google Colab**

```python
# 📂 Cargar archivo
df = pd.read_csv("winequality-red.csv", sep=';')

# 👁️ Vista inicial del dataset
df.head()
```

### 🔹 3. Explorar y entender los datos

```python
# 🧾 Información general
df.info()

# 📊 Distribución de la calidad del vino
sns.countplot(x='quality', data=df)
plt.title("Distribución de calidad del vino 🍷")
plt.xlabel("Calidad (0-10)")
plt.ylabel("Cantidad de muestras")
plt.show()
```

### 🔹 4. Escalar las características

```python
# ✂️ Separar variables numéricas
features = df.drop('quality', axis=1)

# ⚖️ Escalar las características
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
```

### 🔹 5. Aplicar PCA

```python
# 🧠 Aplicar PCA con 2 componentes principales
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_features)

# 📦 Crear nuevo DataFrame con los componentes y la calidad
df_pca = pd.DataFrame(data=pca_components, columns=['PC1', 'PC2'])
df_pca['quality'] = df['quality']
```

### 🔹 6. Visualizar la varianza explicada

```python
# 📈 Porcentaje de varianza explicada por componente
print("Varianza explicada:", pca.explained_variance_ratio_)
print(f"Total de varianza explicada: {sum(pca.explained_variance_ratio_)*100:.2f}%")
```

### 🔹 7. Visualización en 2D con PCA

```python
# 🎨 Visualización PCA por calidad
plt.figure(figsize=(10,6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='quality', palette='coolwarm')
plt.title("Reducción de dimensiones con PCA – Vinos Tintos 🍷")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.grid(True)
plt.legend(title="Calidad")
plt.show()
```

## 🧠 Reflexión

1. ¿Qué tan claramente logra PCA separar los vinos según su calidad?
2. ¿Qué tan útil es PCA si solo vemos una separación parcial?
3. ¿Qué podríamos hacer si queremos mejorar la clasificación con ML?

## 🧾 Conclusión

✅ Aprendiste a cargar un dataset real

✅ Escalaste correctamente los datos

✅ Aplicaste PCA y visualizaste las dimensiones reducidas

✅ Observaste cómo la calidad del vino se agrupa en un espacio más simple

🍇 *"Como un buen vino, los datos también mejoran cuando los destilamos a lo esencial."*