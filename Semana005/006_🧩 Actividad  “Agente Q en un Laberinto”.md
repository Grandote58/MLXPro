# 🧩 **Actividad : “Agente Q en un Laberinto”**

## 🎯 **Objetivos de Aprendizaje**

1. Comprender e implementar el algoritmo **Q-Learning** en entornos estocásticos.
2. Entrenar un agente que mejore su comportamiento a través de la experiencia.
3. Visualizar la evolución del aprendizaje mediante curvas de recompensa.
4. Derivar una **política óptima** a partir de la Q-table.

## 🧰 **Librerías necesarias**

```python
!pip install gymnasium matplotlib numpy --quiet
```

🧰 **Librerías necesarias**

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
from IPython.display import clear_output
```

## 🗂️ **Dataset / Entorno**

Utilizamos el entorno `FrozenLake-v1` del repositorio Gymnasium (Farama Foundation):

 🔗 [Documentación oficial](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)

Este entorno es una grilla de 4x4 donde el agente debe llegar a una meta sin caer en los agujeros.

## 🪜 **Paso a Paso Detallado**

### 🔹 Paso 1: Inicializar el entorno

```python
env = gym.make("FrozenLake-v1", is_slippery=False)
state_space = env.observation_space.n
action_space = env.action_space.n
print(f"🎯 Estados: {state_space} | 🎮 Acciones: {action_space}")
```

### 🔹 Paso 2: Crear la tabla Q

```python
q_table = np.zeros((state_space, action_space))
print("📋 Tabla Q inicial:")
print(q_table)
```

### 🔹 Paso 3: Definir los hiperparámetros

```python
alpha = 0.8        # tasa de aprendizaje
gamma = 0.95       # descuento de futuro
epsilon = 1.0      # nivel de exploración inicial
min_epsilon = 0.01
decay_rate = 0.005
episodes = 1000
max_steps = 100
rewards = []
```

### 🔹 Paso 4: Entrenar al agente

```python
for ep in range(episodes):
    state, _ = env.reset()
    total_reward = 0

    for _ in range(max_steps):
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # exploración
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, _ = env.step(action)

        # Actualización Q-Learning
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action])

        state = next_state
        total_reward += reward

        if terminated or truncated:
            break

    epsilon = max(min_epsilon, epsilon * np.exp(-decay_rate * ep))
    rewards.append(total_reward)
```

### 🔹 Paso 5: Visualizar recompensas por episodio

```python
plt.plot(rewards)
plt.title("📈 Recompensa por episodio")
plt.xlabel("Episodio")
plt.ylabel("Recompensa")
plt.grid(True)
plt.show()
```

### 🔹 Paso 6: Derivar la política aprendida

```python
actions_map = {0: '←', 1: '↓', 2: '→', 3: '↑'}
policy = np.array([actions_map[np.argmax(row)] for row in q_table])
print("🧭 Política aprendida:")
print(policy.reshape(4, 4))
```

## 🎓 **Reflexión**

> “Este ejercicio demuestra cómo un agente puede aprender **por sí mismo** sin conocer previamente el entorno. A través de la exploración y actualización de la tabla Q, el agente construye su **propia estrategia óptima**.
>
> Esta lógica es aplicable a robótica, videojuegos, movilidad urbana e inteligencia autónoma.
>  ¡Y lo mejor es que has construido un agente que tomó decisiones inteligentes desde cero! 🧠🏁”