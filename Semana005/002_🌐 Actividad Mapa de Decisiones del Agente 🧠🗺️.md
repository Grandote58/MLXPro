# **🌐 Actividad: "Mapa de Decisiones del Agente 🧠🗺️"**

## 🎯 Objetivos de Aprendizaje

1. Comprender los componentes fundamentales de un **MDP (Proceso de Decisión de Markov)**.
2. Construir un modelo MDP con estados, acciones, transiciones y recompensas.
3. Visualizar el comportamiento del sistema usando grafos y analizar rutas óptimas.
4. Interpretar cómo las decisiones del agente se ven influenciadas por las probabilidades y recompensas.

## 🧰 Librerías necesarias

```python
!pip install networkx matplotlib pandas --quiet
```

🧰 Librerías necesarias

```python
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
```

## 📁 Dataset

Utilizaremos un dataset real simplificado desde **UCI Machine Learning Repository**:

📄 [Dataset de Navegación Urbana (Urban Navigation Tasks)](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume)

⚠️ En esta actividad, descargamos y preprocesamos una muestra para simular estados urbanos y decisiones posibles para un vehículo autónomo que debe escoger la mejor ruta, representada como un MDP simplificado.

## 🪜 Paso a Paso Detallado

### 🔹 Paso 1: Descargar y preparar datos

```python
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv'
df = pd.read_csv(url, usecols=['holiday', 'weather_main', 'traffic_volume'])
df = df.head(10)  # Solo tomamos una muestra representativa

# Asignamos estados ficticios y acciones posibles
df['estado'] = ['Cruce_A', 'Cruce_B', 'Cruce_C', 'Cruce_D', 'Cruce_E', 'Cruce_F', 'Cruce_G', 'Cruce_H', 'Cruce_I', 'Cruce_J']
df['accion'] = ['Ir_Derecha', 'Esperar', 'Ir_Izquierda', 'Esperar', 'Cruzar', 'Cruzar', 'Ir_Derecha', 'Cruzar', 'Esperar', 'Cruzar']

df = df[['estado', 'accion', 'weather_main', 'traffic_volume']]
df.head()
```

### 🔹 Paso 2: Crear estructura MDP simulada

```
P = {
    'Cruce_A': {'Ir_Derecha': [('Cruce_B', 0.8, 0), ('Cruce_C', 0.2, -1)]},
    'Cruce_B': {'Esperar': [('Cruce_B', 1.0, -0.1)]},
    'Cruce_C': {'Ir_Izquierda': [('Cruce_D', 0.7, 1), ('Cruce_E', 0.3, 0)]},
    'Cruce_D': {'Esperar': [('Cruce_D', 1.0, -0.2)]},
    'Cruce_E': {'Cruzar': [('Cruce_F', 0.5, 1), ('Cruce_G', 0.5, 0)]},
    'Cruce_F': {'Cruzar': [('Cruce_H', 1.0, 0.5)]},
    'Cruce_G': {'Ir_Derecha': [('Cruce_I', 1.0, 0.1)]},
    'Cruce_H': {'Cruzar': [('Cruce_J', 1.0, 2)]},
    'Cruce_I': {'Esperar': [('Cruce_I', 1.0, -0.5)]},
    'Cruce_J': {'Cruzar': [('Cruce_J', 1.0, 0)]}
}
```

### 🔹 Paso 3: Visualizar el MDP como un grafo

```python
G = nx.DiGraph()

# Añadir nodos y aristas
for estado in P:
    for accion in P[estado]:
        for destino, probabilidad, recompensa in P[estado][accion]:
            etiqueta = f"{accion}\nP={probabilidad}, R={recompensa}"
            G.add_edge(estado, destino, label=etiqueta)

# Posiciones automáticas para graficar
pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(12, 8))
nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightgreen')
nx.draw_networkx_labels(G, pos, font_size=10)
edges = G.edges(data=True)
labels = {(u, v): d['label'] for u, v, d in edges}
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8)

plt.title("🚦 Mapa de Decisiones del Agente - Simulación MDP Urbano")
plt.axis('off')
plt.show()
```

## 🎓 Reflexión

> “Cada nodo representa una intersección y cada acción una decisión posible. Como el entorno es incierto (probabilidades distintas), el agente debe aprender qué camino tomar según la probabilidad de éxito y la recompensa acumulada. Esta actividad simula una red vial simplificada para mostrar cómo funciona un MDP en entornos reales como el transporte urbano.”