# 🧪 **Práctica: “EvoCluster: creando clústeres con evolución artificial 🧬”**

### 🎯 **Objetivo**

Utilizar un **Algoritmo Genético (GA)** para **encontrar el número óptimo de clústeres (k)** en un dataset real, evaluando la calidad del agrupamiento con la **métrica de silueta**.

## 📂 **Dataset real utilizado**

- 🧠 **Nombre:** Wholesale customers data
- 🔗 **URL de Kaggle:** https://www.kaggle.com/datasets/vjchoudhary7/wholesale-customers
- 📄 Archivo: `Wholesale customers data.csv`
- 🛒 Contiene datos reales de clientes mayoristas: gastos anuales en productos como leche, detergentes, carnes, etc.

## 👨‍🏫 **Paso a paso en Google Colab (comentado y educativo)**

### 🔹 1. Importar librerías

```python
# 📚 Librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import random
```

### 🔹 2. Cargar el dataset

🔽 **Sube manualmente `Wholesale customers data.csv` desde tu equipo a Colab**

```python
# 📂 Cargar los datos
df = pd.read_csv("Wholesale customers data.csv")
df.head()
```

### 🔹 3. Preprocesamiento

```python
# 🧹 Eliminar columnas no numéricas o irrelevantes
df_clean = df.drop(columns=['Channel', 'Region'])

# ⚖️ Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)
```

### 🔹 4. Función para evaluar la calidad del agrupamiento

```python
# 📏 Función de evaluación: promedio del coeficiente de silueta
def evaluate_k(k):
    if k < 2:
        return -1  # No tiene sentido agrupar en 1
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X_scaled)
    return silhouette_score(X_scaled, labels)
```

### 🔹 5. Definir el Algoritmo Genético

```python
# 🧬 Crear población inicial aleatoria de tamaños de k
def create_population(size, min_k=2, max_k=10):
    return [random.randint(min_k, max_k) for _ in range(size)]

# 🧮 Evaluar población
def evaluate_population(population):
    return [evaluate_k(k) for k in population]

# 🔁 Selección de padres
def select_parents(population, scores):
    sorted_pairs = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
    return [k for k, s in sorted_pairs[:2]]

# 🔄 Cruce
def crossover(parents):
    return (parents[0] + parents[1]) // 2

# 🎲 Mutación
def mutate(k, min_k=2, max_k=10):
    if random.random() < 0.3:
        return random.randint(min_k, max_k)
    return k
```

### 🔹 6. Ejecutar el algoritmo genético

```python
# ⚙️ Parámetros del GA
generations = 10
population_size = 6

# 🧬 Inicialización
population = create_population(population_size)
print("👶 Población inicial:", population)

# 🔁 Evolución
for gen in range(generations):
    scores = evaluate_population(population)
    print(f"🧪 Generación {gen+1}:")
    print(" Población:", population)
    print(" Aptitudes:", scores)

    parents = select_parents(population, scores)
    offspring = [mutate(crossover(parents)) for _ in range(population_size - 2)]

    population = parents + offspring
```

### 🔹 7. Elegir el mejor número de clústeres y visualizar

```python
# 🏁 Seleccionar el mejor k final
final_scores = evaluate_population(population)
best_k = population[np.argmax(final_scores)]
print(f"\n🔍 Mejor número de clústeres encontrado: {best_k}")

# 🎨 Visualización 2D
model = KMeans(n_clusters=best_k)
labels = model.fit_predict(X_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=labels, palette='Set2')
plt.title(f"Agrupamiento final con k={best_k} 🧬")
plt.xlabel("Componente 1 (escalado)")
plt.ylabel("Componente 2 (escalado)")
plt.grid(True)
plt.show()
```

## 🧠 Reflexión (e-learning)

1. ¿Qué tan diferente fue el k óptimo encontrado por el algoritmo genético comparado con una elección manual?
2. ¿Qué ventaja tiene un GA frente a una búsqueda exhaustiva?
3. ¿Qué efecto tiene la mutación en la exploración del espacio de soluciones?

## 📎 Conclusión

✅ Usaste un **algoritmo inspirado en la evolución**

✅ Optimizaste un modelo no supervisado usando silueta

✅ Visualizaste resultados reales de segmentación de clientes

✅ Aprendiste cómo la inteligencia evolutiva ayuda a resolver problemas complejos

🔬 *“No siempre se trata de tener una fórmula exacta… a veces la evolución tiene la última palabra.”*