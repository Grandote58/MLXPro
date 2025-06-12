# **🧭 Actividad : “Valorando el Camino del Explorador”**

## 🎯 Objetivos de Aprendizaje

1. Comprender los conceptos de **política**, **valor de estado** y **valor de acción** en un entorno realista.
2. Aplicar estos conceptos utilizando un **dataset real** como base para modelar decisiones secuenciales.
3. Visualizar las funciones de valor con gráficas interpretables.
4. Desarrollar un pensamiento crítico sobre decisiones óptimas bajo incertidumbre.

## 🧰 Librerías necesarias

```python
!pip install pandas matplotlib seaborn --quiet
```

🧰 Librerías necesarias

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
```

## 🗂️ Dataset utilizado

📄 Dataset: **Metro Interstate Traffic Volume**
 🔗 [UCI Repository](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume)
 📥 Descarga directa:

```python
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv"
```

Este dataset contiene datos del tráfico urbano. Utilizaremos el volumen de tráfico para simular un entorno secuencial donde un agente debe decidir entre “esperar” o “avanzar”.

## 🪜 Paso a Paso Detallado

### 🔹 Paso 1: Cargar y preparar los datos

```python
df = pd.read_csv(url)
df_sample = df[['holiday', 'traffic_volume']].head(10).copy()
df_sample['state'] = [f'S{i}' for i in range(len(df_sample))]
df_sample.reset_index(drop=True, inplace=True)
df_sample
```

### 🔹 Paso 2: Definir la política simple (π) y recompensas

```python
# Política simple basada en tráfico
df_sample['action'] = df_sample['traffic_volume'].apply(lambda x: 'avanzar' if x < 4000 else 'esperar')

# Recompensa simulada: +1 por avanzar con poco tráfico, -1 por avanzar con mucho tráfico, 0 por esperar
def calcular_recompensa(row):
    if row['action'] == 'avanzar':
        return 1 if row['traffic_volume'] < 4000 else -1
    else:
        return 0

df_sample['reward'] = df_sample.apply(calcular_recompensa, axis=1)
df_sample[['state', 'action', 'reward']]
```

### 🔹 Paso 3: Calcular la función de valor V(s)

```python
gamma = 0.9  # factor de descuento
df_sample['V'] = 0.0

# Se calcula hacia atrás para simular planificación
for i in reversed(range(len(df_sample))):
    r = df_sample.loc[i, 'reward']
    v_next = df_sample.loc[i+1, 'V'] if i+1 < len(df_sample) else 0
    df_sample.loc[i, 'V'] = r + gamma * v_next

df_sample[['state', 'action', 'reward', 'V']]
```

### 🔹 Paso 4: Calcular la función de acción-valor Q(s,a)

```python
Q = []
for i in range(len(df_sample)):
    s = df_sample.loc[i, 'state']
    for a in ['avanzar', 'esperar']:
        if a == 'esperar':
            r = 0
            v = gamma * df_sample.loc[i, 'V']
        else:
            r = 1 if df_sample.loc[i, 'traffic_volume'] < 4000 else -1
            v = gamma * df_sample.loc[i+1, 'V'] if i+1 < len(df_sample) else 0
        Q.append({'state': s, 'action': a, 'Q_value': r + v})

df_q = pd.DataFrame(Q)
df_q.pivot(index='state', columns='action', values='Q_value')
```

### 🔹 Paso 5: Visualizar con un heatmap 🧯

```python
plt.figure(figsize=(10,4))
pivot_q = df_q.pivot(index='state', columns='action', values='Q_value')
sns.heatmap(pivot_q, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("🔢 Función Q(s,a): Valor esperado por acción")
plt.show()
```

## 🎓 Reflexión 

> “Has usado datos reales para calcular cuánto vale cada estado y cada acción posible en él. ¡Eso es inteligencia artificial aplicada! Esta es la base que usan sistemas autónomos para actuar de forma óptima en entornos reales. 🚦🤖”