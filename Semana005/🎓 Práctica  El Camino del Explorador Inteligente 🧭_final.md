# 🎓 **Práctica : El Camino del Explorador Inteligente 🧭**

## 🎯 **Objetivo General**

Guiar paso a paso en la comprensión del equilibrio entre **exploración** y **explotación** usando un entorno visual y experimental de aprendizaje por reforzamiento.

## 🧠 **Contexto de Aprendizaje**

Imagina que eres un explorador 🧭 en una isla misteriosa. Tienes que encontrar la salida (la meta 🏁) evitando caer en trampas 🕳️. Solo sabrás si un camino es bueno al probarlo… pero no puedes perder tiempo tomando malas decisiones.

Así funciona el **Q-Learning**: tu agente irá aprendiendo cuáles decisiones tomar explorando al principio y explotando su conocimiento cuando esté más entrenado.

## 📦 **Librerías necesarias**

```python
!pip install gymnasium numpy matplotlib --quiet

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
```

## 🧪 **Entorno**

Usaremos el entorno `FrozenLake-v1` de Gymnasium (4x4, sin resbalones), ideal para visualizar decisiones en un entorno cuadrado.

```python
env = gym.make("FrozenLake-v1", is_slippery=False)
state_space = env.observation_space.n
action_space = env.action_space.n
print(f"🌐 Estados: {state_space}, 🎮 Acciones posibles: {action_space}")
```

## 🗺️ **Paso 1: Inicializar Q-Table y parámetros**

```python
q_table = np.zeros((state_space, action_space))

# Hiperparámetros
alpha = 0.8          # tasa de aprendizaje
gamma = 0.95         # descuento de recompensa futura
epsilon = 1.0        # nivel de exploración inicial
min_epsilon = 0.01
decay_rate = 0.005

episodes = 500
max_steps = 100

rewards = []
epsilon_values = []
```

## 🤖 **Paso 2: Entrenar al Explorador Q**

```python
for ep in range(episodes):
    state, _ = env.reset()
    total_reward = 0

    for _ in range(max_steps):
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # 🔎 Exploración
        else:
            action = np.argmax(q_table[state])  # 🧠 Explotación

        new_state, reward, terminated, truncated, _ = env.step(action)
        q_table[state, action] += alpha * (reward + gamma * np.max(q_table[new_state]) - q_table[state, action])
        state = new_state
        total_reward += reward

        if terminated or truncated:
            break

    epsilon = max(min_epsilon, epsilon * np.exp(-decay_rate * ep))
    epsilon_values.append(epsilon)
    rewards.append(total_reward)
```

## 📊 **Paso 3: Visualización**

### 🔁 Evolución de Epsilon (ε)

```python
plt.figure(figsize=(10, 4))
plt.plot(epsilon_values, color='orange')
plt.title("📉 Evolución del valor de ε (Exploración)")
plt.xlabel("Episodios")
plt.ylabel("Valor de ε")
plt.grid(True)
plt.show()
```

### 🏆 Recompensas por episodio

```python
plt.figure(figsize=(10, 4))
plt.plot(rewards, color='green')
plt.title("🏆 Recompensas acumuladas por episodio")
plt.xlabel("Episodios")
plt.ylabel("Recompensa")
plt.grid(True)
plt.show()
```

## 🧭 **Paso 4: Visualizar la Política Aprendida**

```python
actions_map = {0: '←', 1: '↓', 2: '→', 3: '↑'}
policy = np.array([actions_map[np.argmax(row)] for row in q_table])
print("🧭 Política final aprendida:")
print(policy.reshape(4, 4))
```

## 🎓 **Reflexión**

> ✅ Al principio, el agente **explora** mucho porque no tiene información.
>
> ✅ Con el tiempo, **explotar** le da mejores resultados porque confía en su experiencia.
>
> 🧭 ¡Así se entrena la inteligencia! Aprender a decidir cuándo arriesgar y cuándo actuar con certeza.

📌 **Preguntas para reforzar** :

- ¿Qué pasaría si nunca disminuyéramos el valor de ε?
- ¿Qué observas en la política aprendida?
- ¿Podrías adaptar esto a una situación real como un GPS o un robot móvil?