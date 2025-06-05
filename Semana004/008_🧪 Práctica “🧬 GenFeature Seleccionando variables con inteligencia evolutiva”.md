# 🧪 **Práctica: “🧬 GenFeature: Seleccionando variables con inteligencia evolutiva”**

### 🎯 **Objetivo**

Aplicar un **Algoritmo Genético** para realizar **selección de características** (feature selection) sobre un dataset real, con el fin de **optimizar el rendimiento de un clasificador** (Random Forest). Se usará la puntuación F1 para evaluar la calidad de las soluciones.

## 📂 **Dataset real utilizado**

- 🧠 **Nombre:** Breast Cancer Wisconsin (Diagnostic) Data Set
- 🔗 **URL de Kaggle:**
   👉 https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
- 📄 Archivo: `data.csv`
- 📊 Incluye 569 registros de pacientes con 30 características, clasificadas como **malignas o benignas**

## 👨‍🏫 **Paso a paso en Google Colab (explicado y con emojis)**

### 🔹 1. Importar librerías necesarias

```python
# 📚 Librerías para análisis, ML y evolución
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
```

### 🔹 2. Cargar el dataset

🔽 **Sube `data.csv` desde Kaggle a Google Colab**

```python
# 📂 Cargar datos
df = pd.read_csv('data.csv')
df = df.drop(columns=['id', 'Unnamed: 32'])  # 🧽 Eliminar columnas irrelevantes
df.head()
```

### 🔹 3. Preparar datos

```python
# 🎯 Variables predictoras y objetivo
X = df.drop('diagnosis', axis=1)
y = df['diagnosis'].map({'M': 1, 'B': 0})  # Convertimos a binario

# 🧪 División de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 🔹 4. Función de evaluación (fitness)

```python
# 🧬 Evaluar una solución (subset de features)
def evaluate(individual):
    selected = [i for i, bit in enumerate(individual) if bit == 1]
    if not selected: return 0  # ⚠️ Si no se elige nada, score 0

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train.iloc[:, selected], y_train)
    preds = model.predict(X_test.iloc[:, selected])
    return f1_score(y_test, preds)
```

### 🔹 5. Crear algoritmo genético

```python
# 🔧 Parámetros básicos
num_features = X.shape[1]
population_size = 10
generations = 20

# 🔢 Crear población inicial
def create_population():
    return [np.random.randint(0, 2, num_features).tolist() for _ in range(population_size)]

# 💡 Selección de padres (los dos mejores)
def select_parents(population, scores):
    sorted_pairs = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
    return [sorted_pairs[0][0], sorted_pairs[1][0]]

# 🔀 Cruce simple
def crossover(p1, p2):
    point = random.randint(1, num_features - 1)
    return p1[:point] + p2[point:]

# 🎲 Mutación aleatoria
def mutate(individual, prob=0.1):
    return [bit if random.random() > prob else 1 - bit for bit in individual]
```

### 🔹 6. Ejecutar la evolución

```python
# 🧪 Iniciar población
population = create_population()
best_scores = []

for gen in range(generations):
    scores = [evaluate(ind) for ind in population]
    best_scores.append(max(scores))
    
    print(f"🧬 Generación {gen+1} | Mejor F1: {max(scores):.4f}")
    
    parents = select_parents(population, scores)
    offspring = [mutate(crossover(parents[0], parents[1])) for _ in range(population_size - 2)]
    population = parents + offspring
```

### 🔹 7. Visualizar evolución del rendimiento

```python
plt.plot(range(1, generations+1), best_scores, marker='o')
plt.title("📈 Evolución del Mejor F1-score por Generación")
plt.xlabel("Generación")
plt.ylabel("F1-score")
plt.grid(True)
plt.show()
```

## 🧠 Reflexión

1. ¿Qué tan relevante fue reducir las características?
2. ¿Qué efecto tiene el tamaño de la población en los resultados?
3. ¿Cómo influye la tasa de mutación en la exploración del espacio?

## 📎 Conclusión

✅ Aplicaste un Algoritmo Genético en un caso real

✅ Seleccionaste características de forma automática

✅ Aumentaste el rendimiento del modelo reduciendo variables

✅ Entendiste cómo cruzar y mutar soluciones evolutivas

🧠 *“Los datos también evolucionan... cuando tú los guías con inteligencia.”*