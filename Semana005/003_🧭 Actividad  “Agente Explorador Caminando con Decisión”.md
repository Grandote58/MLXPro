# 🧭 **Actividad : “Agente Explorador: Caminando con Decisión”**

## 🎯 **Objetivos de Aprendizaje**

1. Comprender el **ciclo agente-ambiente** mediante un modelo de exploración secuencial.
2. Implementar desde cero un agente que interactúa con un entorno basado en estados.
3. Simular episodios de interacción y observar los cambios en tiempo real.
4. Utilizar datos reales para alimentar decisiones en el entorno.

## 🧰 **Librerías necesarias**

```python
!pip install pandas matplotlib --quiet
```

🧰 **Librerías necesarias**

```python
import pandas as pd
import matplotlib.pyplot as plt
import time
```

## 🗂️ **Dataset real a utilizar**

Usaremos un subconjunto del dataset **"Metro Interstate Traffic Volume"** de UCI para simular la influencia del tráfico en las decisiones del agente (avanzar o esperar).

📦 [Descarga directa CSV](https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv)
 📄 [Página del dataset UCI](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume)

## 🪜 **Paso a Paso detallado**

### 🔹 **Paso 1: Cargar y preparar el entorno con datos reales**

```python
# Cargar los primeros 10 estados desde el dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv"
df = pd.read_csv(url)

# Filtramos los primeros 10 registros útiles para crear un entorno simplificado
df_sample = df[['holiday', 'weather_main', 'traffic_volume']].head(10)
df_sample['state'] = [f'Estado_{i}' for i in range(10)]
df_sample.reset_index(drop=True, inplace=True)

# Mostrar el entorno
df_sample
```

### 🔹 **Paso 2: Definir la clase del entorno**

```python
class TrafficEnv:
    def __init__(self, data):
        self.data = data
        self.state_index = 0
        self.max_state = len(data) - 1

    def step(self, action):
        state_data = self.data.iloc[self.state_index]
        traffic = state_data['traffic_volume']
        
        if action == "avanzar" and traffic < 4000:
            reward = 1
            self.state_index = min(self.state_index + 1, self.max_state)
        elif action == "esperar":
            reward = 0.5
        else:
            reward = -1

        done = self.state_index == self.max_state
        return self.state_index, reward, done

    def reset(self):
        self.state_index = 0
        return self.state_index
```

### 🔹 **Paso 3: Simular al agente en múltiples episodios**

```python
env = TrafficEnv(df_sample)

episodios = 5
historial = []

for ep in range(episodios):
    estado = env.reset()
    print(f"\n🎬 Episodio {ep+1}")
    pasos = 0
    while True:
        estado_data = df_sample.iloc[estado]
        trafico = estado_data['traffic_volume']
        accion = "avanzar" if trafico < 4000 else "esperar"
        
        nuevo_estado, recompensa, terminado = env.step(accion)
        historial.append((ep+1, estado, accion, nuevo_estado, recompensa))
        print(f"📍 Estado: {estado} | 🚦 Tráfico: {trafico} | ⚙️ Acción: {accion} | 🏅 Recompensa: {recompensa}")
        
        estado = nuevo_estado
        pasos += 1
        time.sleep(0.3)
        
        if terminado or pasos > 15:
            print("✅ Fin del episodio")
            break
```

### 🔹 **Paso 4: Visualizar desempeño**

```python
# Visualización de la cantidad de pasos por episodio
pasos_ep = {}
for ep, _, _, _, _ in historial:
    pasos_ep[ep] = pasos_ep.get(ep, 0) + 1

plt.bar(pasos_ep.keys(), pasos_ep.values(), color='lightgreen')
plt.title("📊 Pasos por episodio hasta el objetivo")
plt.xlabel("Episodio")
plt.ylabel("Pasos")
plt.grid(True)
plt.show()
```

## 🎓 **Reflexión**

> “El agente en esta simulación aprende a leer el entorno y decidir si avanzar o esperar según las condiciones del tráfico real. Este es un modelo básico pero poderoso del ciclo agente-ambiente. Es la base que luego se refina con aprendizaje Q, funciones de valor y aprendizaje profundo.”