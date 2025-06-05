# 🧪 Práctica Avanzada: *“ClusterMall 🛍️: Segmentando Clientes con K-Means”*

### 🎯 **Objetivo**

Aplicar el algoritmo de *K-Means Clustering* a un conjunto real de datos de clientes de un centro comercial. El propósito es segmentarlos en grupos con comportamientos similares basados en edad, género e ingresos.

### 📂 **📥 Dataset Real**

**Nombre del dataset:** *Mall Customers Dataset*
 **Fuente:** Kaggle
 **Enlace de descarga directa:**

 🔗 https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial

> Si estás en Google Colab, primero debes subir el archivo `Mall_Customers.csv` desde tu PC con el botón “Subir archivos” (📁 ícono en la barra lateral izquierda).



## 👨‍🏫 **Paso a paso con explicación y emojis**

### 🔹 Paso 1: Importar librerías

```python
# 🧠 Librerías para manipulación de datos y visualización
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
```

### 🔹 Paso 2: Cargar el dataset

```python
# 📂 Cargar archivo local (subido previamente en Colab)
df = pd.read_csv('Mall_Customers.csv')
df.head()
```

**🎯 Nota:** El dataset contiene:

- CustomerID
- Genre
- Age
- Annual Income (k$)
- Spending Score (1-100)

### 🔹 Paso 3: Análisis exploratorio

```python
# 👁️‍🗨️ Ver estructura general
df.info()

# 📊 Estadísticas básicas
df.describe()

# 📌 Visualización por género
sns.countplot(x='Genre', data=df)
plt.title("Distribución por Género 🧑‍🤝‍🧑")
plt.show()
```

### 🔹 Paso 4: Preparar los datos

```python
# ✂️ Eliminar columnas que no aportan al clustering
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# 🧽 Escalado de datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 🔹 Paso 5: Elegir número óptimo de clusters (Método del codo)

```python
# 📐 Método del codo
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# 📊 Gráfico del codo
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Método del Codo 💪')
plt.xlabel('Número de Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()
```

### 🔹 Paso 6: Aplicar K-Means con el número óptimo

```python
# 🚀 Elegir k = 5 (ejemplo basado en el gráfico)
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
```

### 🔹 Paso 7: Visualizar clústeres

```python
# 🖼️ Gráfico 2D usando PCA opcionalmente o Age vs Income
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Cluster', data=df, palette='Set2', s=100)
plt.title('Clusters de Clientes - ClusterMall 🛍️')
plt.xlabel('Ingreso Anual (k$)')
plt.ylabel('Puntaje de Gasto')
plt.legend(title='Segmento')
plt.grid(True)
plt.show()
```

### 🧠 Preguntas de reflexión (e-learning)

1. ¿Qué características parecen definir mejor los clústeres?
2. ¿Cuál sería una campaña de marketing ideal para cada grupo?
3. ¿Qué otras variables te gustaría agregar para mejorar la segmentación?

### 📎 Conclusión

Esta práctica muestra cómo aplicar K-Means a un caso real de segmentación de clientes. Usamos análisis exploratorio, preprocesamiento, método del codo y visualización para entender cómo crear valor a partir de datos reales. 🧠📈