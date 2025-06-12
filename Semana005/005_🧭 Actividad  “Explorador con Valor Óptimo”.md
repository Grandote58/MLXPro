# 🧭 **Actividad : “Explorador con Valor Óptimo”**

## 🎯 **Objetivos de Aprendizaje**

1. Comprender el método de **Value Iteration** aplicado a problemas de decisión secuencial.
2. Implementar el cálculo de valores óptimos y derivar la **política óptima**.
3. Utilizar un dataset real como base para simular decisiones de un agente.
4. Visualizar la **convergencia del algoritmo** y la estrategia del agente.

## 🧰 **Librerías necesarias**

```python
!pip install pandas numpy matplotlib --quiet
```

🧰 **Librerías necesarias**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

## 📦 **Dataset real utilizado**

📄 Dataset: [Metro Interstate Traffic Volume – UCI](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume)

Este dataset contiene registros reales del volumen de tráfico. Lo usaremos para modelar un entorno tipo línea de decisión (MDP secuencial), donde el agente debe decidir si **avanza** o **espera** en función del tráfico y obtener la política óptima con **Value Iteration**.

## 🪜 **Paso a Paso Detallado**

### 🔹 Paso 1: Cargar y preparar los datos

```python
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv"
df = pd.read_csv(url)
df_sample = df[['traffic_volume']].head(10).copy()
df_sample['state'] = [f"S{i}" for i in range(len(df_sample))]
df_sample.reset_index(drop=True, inplace=True)
df_sample
```

### 🔹 Paso 2: Inicializar las variables del modelo MDP

```python
states = df_sample['state'].tolist()
actions = ['avanzar', 'esperar']
gamma = 0.9
threshold = 1e-4

# Inicialización de valores y recompensas
V = {s: 0 for s in states}
rewards = {}

for i, row in df_sample.iterrows():
    if row['traffic_volume'] < 4000:
        rewards[row['state']] = 1  # recompensa por avanzar con tráfico bajo
    elif row['traffic_volume'] < 6000:
        rewards[row['state']] = 0  # tráfico medio, sin penalización
    else:
        rewards[row['state']] = -1  # penalización por tráfico alto
```

### 🔹 Paso 3: Implementar Value Iteration

```python
deltas = []
iterations = 0

while True:
    delta = 0
    new_V = {}
    for i, s in enumerate(states):
        q_sa = []
        for a in actions:
            if a == 'esperar':
                r = -0.2  # costo de esperar
                next_state = s
            else:
                r = rewards[s]
                next_state = states[i+1] if i+1 < len(states) else s
            q = r + gamma * V[next_state]
            q_sa.append(q)
        best_q = max(q_sa)
        new_V[s] = best_q
        delta = max(delta, abs(V[s] - best_q))
    V = new_V
    deltas.append(delta)
    iterations += 1
    if delta < threshold:
        break

print(f"✅ Convergencia en {iterations} iteraciones")
```

### 🔹 Paso 4: Derivar la política óptima

```python
policy = {}
for i, s in enumerate(states):
    q_sa = {}
    for a in actions:
        if a == 'esperar':
            r = -0.2
            next_state = s
        else:
            r = rewards[s]
            next_state = states[i+1] if i+1 < len(states) else s
        q_sa[a] = r + gamma * V[next_state]
    policy[s] = max(q_sa, key=q_sa.get)

# Mostrar política
pd.DataFrame.from_dict(policy, orient='index', columns=['Mejor Acción'])
```

### 🔹 Paso 5: Visualizar la convergencia del algoritmo

```python
plt.plot(deltas)
plt.title("📉 Convergencia de Value Iteration")
plt.xlabel("Iteraciones")
plt.ylabel("Delta (máxima diferencia)")
plt.grid(True)
plt.show()
```

## 🎓 **Reflexión **

> 🧠 En esta actividad, el agente explorador aprendió a **valorar estados** y tomar decisiones óptimas con base en datos reales.
>  🔍 Value Iteration permitió calcular el mejor camino en función del tráfico y el costo de esperar.
>  🛣️ Este modelo se puede escalar a mapas urbanos, planificación logística y sistemas de movilidad inteligente.