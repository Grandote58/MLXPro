# **🧠 Actividad: "Mi Primer Agente Inteligente: ¡Rumbo a la Meta! 🎯"**

### 👩‍🎓 Dirigido a:

Estudiantes de pregrado y posgrado en cursos de Introducción al Machine Learning o Inteligencia Artificial.

## 🎯 Objetivos de Aprendizaje

1. Comprender los fundamentos del **Aprendizaje por Reforzamiento (RL)** y la interacción *agente ↔ ambiente*.
2. Implementar un agente simple con el algoritmo **Q-Learning**.
3. Analizar la evolución del agente a través de recompensas acumuladas.
4. Interpretar los resultados visualmente y ajustar parámetros del aprendizaje.

## 🛠️ Librerías necesarias

```python
!pip install gymnasium[classic_control] --quiet
```

### 🛠️ Librerías necesarias

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from IPython.display import clear_output
```

## 📁 Dataset Utilizado

Utilizaremos el entorno **FrozenLake-v1** del paquete `Gymnasium`, un ambiente estándar en RL.

 📄 **Repositorio oficial:** https://gymnasium.farama.org/environments/toy_text/frozen_lake/

Este entorno representa un lago congelado donde el agente debe **llegar a una meta** sin caer en los agujeros.

## 🧪 Paso a Paso Detallado

### 🔹 Paso 1: Crear el entorno

```python
env = gym.make("FrozenLake-v1", is_slippery=False)
state_space = env.observation_space.n
action_space = env.action_space.n

print(f"🔢 Estados posibles: {state_space}")
print(f"🎮 Acciones posibles: {action_space}")
```

> ✅ `is_slippery=False` hace que el entorno sea determinista, facilitando el aprendizaje inicial.

### 🔹 Paso 2: Inicializar la tabla Q

```python
q_table = np.zeros((state_space, action_space))
print("📊 Tabla Q inicializada:")
print(q_table)
```

### 🔹 Paso 3: Definir los hiperparámetros

```python
# Número de episodios de entrenamiento
episodes = 1000
max_steps = 100

# Hiperparámetros de Q-Learning
learning_rate = 0.8
discount_factor = 0.95

# Exploración vs Explotación
epsilon = 1.0       # exploración inicial
min_epsilon = 0.01
decay_rate = 0.005
```

### 🔹 Paso 4: Entrenamiento del agente

```python
rewards = []

for episode in range(episodes):
    state, _ = env.reset()
    total_rewards = 0

    for step in range(max_steps):
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # exploración
        else:
            action = np.argmax(q_table[state])  # explotación

        next_state, reward, terminated, truncated, _ = env.step(action)

        q_table[state, action] += learning_rate * (reward + discount_factor * np.max(q_table[next_state]) - q_table[state, action])

        state = next_state
        total_rewards += reward

        if terminated or truncated:
            break

    epsilon = max(min_epsilon, epsilon * np.exp(-decay_rate * episode))
    rewards.append(total_rewards)

print("✅ Entrenamiento finalizado")
```

### 🔹 Paso 5: Visualizar recompensas por episodio

```python
plt.plot(rewards)
plt.title("🎯 Recompensas por episodio")
plt.xlabel("Episodios")
plt.ylabel("Recompensa")
plt.grid(True)
plt.show()
```

### 🔹 Paso 6: Probar el agente entrenado

```python
for i in range(3):
    state, _ = env.reset()
    print(f"\n🚀 Episodio {i+1}")
    time.sleep(1)

    for step in range(max_steps):
        clear_output(wait=True)
        env.render()
        time.sleep(0.5)

        action = np.argmax(q_table[state])
        state, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            clear_output(wait=True)
            env.render()
            if reward == 1:
                print("🎉 ¡El agente alcanzó la meta!")
            else:
                print("💧 ¡El agente cayó en el agua!")
            time.sleep(2)
            break
env.close()
```

## 🎓 Reflexión Final



> 📌 “Este ejercicio muestra cómo un agente sin conocimiento previo puede aprender a tomar decisiones óptimas simplemente interactuando con su entorno. Usando Q-Learning, el agente ajusta su comportamiento con base en recompensas, un paso crucial hacia sistemas inteligentes autónomos.”

