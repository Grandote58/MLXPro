# 🧠 Actividad eLearning: **“El Explorador Curioso 🧭”**

### 🎯 Objetivos de Aprendizaje

- Comprender el dilema **exploración vs. explotación** en Aprendizaje por Reforzamiento.
- Implementar el algoritmo **ε-greedy** y observar su impacto en el rendimiento del agente.
- Visualizar cómo el valor de ε\varepsilonε afecta la toma de decisiones.
- Utilizar un entorno realista y visual de un laberinto con obstáculos.

### 📦 Librerías necesarias

```python
!pip install gymnasium matplotlib numpy --quiet
```

📦 Librerías necesarias

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
```

### 🗺️ Dataset / Entorno utilizado

Utilizaremos el entorno educativo `FrozenLake-v1` (modo sin resbalones) del repositorio **Farama Gymnasium**:

📦 El entorno simula un lago congelado con agujeros, y el agente debe llegar a la meta sin caer.

## 🪜 Paso a Paso Detallado

### 🔹 Paso 1: Inicializar entorno y Q-table

```python
env = gym.make("FrozenLake-v1", is_slippery=False)
state_space = env.observation_space.n
action_space = env.action_space.n

print(f"Estados: {state_space} | Acciones: {action_space}")
q_table = np.zeros((state_space, action_space))
```

### 🔹 Paso 2: Definir hiperparámetros y estructuras

```python
alpha = 0.8        # tasa de aprendizaje
gamma = 0.95       # descuento de futuro
epsilon = 1.0      # valor inicial de exploración
min_epsilon = 0.01
decay_rate = 0.005
episodes = 500
max_steps = 100

rewards = []
epsilon_values = []
```

### 🔹 Paso 3: Entrenar el agente con estrategia ε-greedy

```python
for ep in range(episodes):
    state, _ = env.reset()
    total_reward = 0

    for _ in range(max_steps):
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # 🔎 Explora
        else:
            action = np.argmax(q_table[state])  # 🧠 Explota

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

### 🔹 Paso 4: Visualizar cómo evoluciona la exploración

```python
plt.plot(epsilon_values)
plt.title("📉 Evolución del valor de ε (Epsilon)")
plt.xlabel("Episodios")
plt.ylabel("Exploración")
plt.grid(True)
plt.show()
```

### 🔹 Paso 5: Visualizar el aprendizaje del agente

```python
plt.plot(rewards)
plt.title("🏆 Recompensas acumuladas por episodio")
plt.xlabel("Episodios")
plt.ylabel("Recompensa")
plt.grid(True)
plt.show()
```

### 🔹 Paso 6: Política aprendida del agente

```python
actions_map = {0: '←', 1: '↓', 2: '→', 3: '↑'}
policy = np.array([actions_map[np.argmax(row)] for row in q_table])
print("🧭 Política aprendida:")
print(policy.reshape(4, 4))
```

### 🎓 Reflexión

> “Un agente inteligente aprende a **equilibrar entre probar lo desconocido y usar lo aprendido**.
>  Gracias al algoritmo ε-greedy, el Explorador Curioso va disminuyendo su curiosidad a medida que mejora su conocimiento.
>
>  Este principio es usado en robots autónomos, videojuegos y sistemas de recomendación.
>  ¡Hoy tú también has aprendido a equilibrar la exploración con sabiduría! 🚀”