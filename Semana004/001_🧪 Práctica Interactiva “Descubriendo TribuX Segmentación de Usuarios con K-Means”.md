# 🧪 Práctica Interactiva: *“Descubriendo TribuX: Segmentación de Usuarios con K-Means”*

### 🎯 **Objetivo**

Aplicar el algoritmo de *K-Means Clustering* para segmentar perfiles de usuarios a partir de características como edad, ingresos y frecuencia de compra. Se visualizarán los resultados en gráficos 2D para comprender la composición de cada grupo.

### 🛠️ **Herramientas**

- Google Colab
- Librerías: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`
- Dataset simulado en tiempo real

### 👨‍🏫 **Paso a Paso en Google Colab**

#### 📌 1. Preparar el entorno

```python
# 🚀 Cargar librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
```

#### 📌 2. Crear el dataset de usuarios “TribuX”

```python
# 🧪 Simular datos de usuarios
np.random.seed(42)
data = {
    'Edad': np.random.randint(18, 60, 100),
    'Ingresos_Mensuales': np.random.randint(1000, 10000, 100),
    'Frecuencia_Compra_Mensual': np.random.randint(1, 30, 100)
}

df = pd.DataFrame(data)
df.head()
```

#### 📌 3. Visualizar los datos

```python
# 🔍 Análisis exploratorio básico
sns.pairplot(df)
plt.suptitle("Relaciones entre variables de usuarios", y=1.02)
plt.show()
```

#### 📌 4. Escalar los datos

```python
# ⚖️ Estandarizar los datos para mejorar el rendimiento de K-Means
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
```

#### 📌 5. Aplicar K-Means

```python
# 📊 Definir número de clusters
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)
```

#### 📌 6. Visualizar los clústeres

```python
# 🎨 Visualización de clusters en 2D
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='Ingresos_Mensuales', y='Frecuencia_Compra_Mensual', hue='Cluster', palette='Set2')
plt.title("Segmentación de usuarios - TribuX 💡")
plt.xlabel("Ingresos Mensuales")
plt.ylabel("Frecuencia de Compra")
plt.legend(title="Cluster")
plt.grid(True)
plt.show()
```

#### 📌 7. Interpretación de resultados

```python
# 📊 Promedios por clúster para interpretar segmentos
df.groupby('Cluster').mean()
```

### 📝 **Preguntas para Reflexión (e-learning)**

1. ¿Qué diferencias observas entre los grupos formados?
2. ¿Cómo podrían estos segmentos ser útiles para una estrategia de marketing?
3. ¿Qué pasaría si usamos más (o menos) clusters?

### 📎 **Conclusión de la práctica**

Con K-Means descubrimos tres tribus ocultas dentro del comportamiento de usuarios de una tienda.

Estas agrupaciones permiten diseñar campañas personalizadas, mejorar la atención al cliente y tomar decisiones basadas en datos.