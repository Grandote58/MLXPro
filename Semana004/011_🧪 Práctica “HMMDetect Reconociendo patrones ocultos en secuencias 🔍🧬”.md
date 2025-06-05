# 🧪 **Práctica: “HMMDetect: Reconociendo patrones ocultos en secuencias 🔍🧬”**

### 🎯 **Objetivo**

Aplicar un **Modelo Oculto de Markov (HMM)** sobre un dataset real de actividad humana para **identificar patrones ocultos en secuencias temporales**, utilizando Python y la librería `hmmlearn`.

## 📂 **Dataset real utilizado**

- **📘 Nombre:** Human Activity Recognition Using Smartphones
- 🌐 **URL de Kaggle:**
   👉 https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones
- 📄 Archivo: `train/X_train.txt` y `train/y_train.txt`
- 👣 Contiene sensores de acelerómetro y giroscopio mientras usuarios caminan, suben escaleras, se sientan, etc.

## 👨‍🏫 **Paso a paso en Google Colab (con emojis y explicación)**

### 🔹 1. Instalar librerías necesarias

```python
# 📦 Instalar hmmlearn si no está instalada
!pip install hmmlearn --quiet
```

### 🔹 2. Importar librerías

```python
# 📚 Librerías de modelado y análisis
import numpy as np
import pandas as pd
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns
```

### 🔹 3. Cargar el dataset

🔽 **Sube `X_train.txt` y `y_train.txt` manualmente desde el dataset de Kaggle**

```python
# 📂 Cargar datos de características y etiquetas
X = pd.read_csv("X_train.txt", delim_whitespace=True, header=None)
y = pd.read_csv("y_train.txt", delim_whitespace=True, header=None)

# 👀 Vista previa de los datos
print("📐 Tamaño de X:", X.shape)
print("🔖 Clases de actividad (y):", y[0].unique())
```

### 🔹 4. Preprocesar los datos

```python
# ⚙️ Usaremos solo algunas columnas para simplificar
X_simple = X.iloc[:, :3]  # Tomamos solo 3 dimensiones para visualizar

# 📊 Convertimos a arreglo NumPy
X_np = X_simple.to_numpy()

# 🧮 Longitud de cada secuencia: cada 128 filas = una muestra
lengths = [128] * int(X_np.shape[0] / 128)
```

### 🔹 5. Entrenar el modelo HMM

```python
# 🧠 Entrenar un HMM con 6 estados ocultos (uno por tipo de actividad esperada)
model = hmm.GaussianHMM(n_components=6, covariance_type="diag", n_iter=100, random_state=42)
model.fit(X_np, lengths)

print("✅ Modelo entrenado correctamente")
```

### 🔹 6. Predecir estados ocultos

```python
# 🔍 Obtener los estados ocultos predichos
hidden_states = model.predict(X_np, lengths)

# 📊 Mostrar los primeros 20 estados
print("🔐 Estados ocultos (primeros 20):", hidden_states[:20])
```

### 🔹 7. Visualizar los estados ocultos

```python
# 🎨 Visualizamos cómo varían los estados ocultos en el tiempo
plt.figure(figsize=(15, 3))
plt.plot(hidden_states, color='purple')
plt.title("Secuencia de estados ocultos inferidos 🔎")
plt.xlabel("Tiempo (muestras concatenadas)")
plt.ylabel("Estado oculto")
plt.grid(True)
plt.show()
```

### 🔹 8. Comparar con etiquetas reales

```python
# 🔁 Repetimos etiquetas para alinear con datos por fila
real_labels = y[0].repeat(128).reset_index(drop=True)

# 🧱 Mostrar comparación
comparison = pd.DataFrame({'Estado_oculto': hidden_states, 'Actividad_real': real_labels})
comparison.head(10)
```

## 🧠 Reflexión

1. ¿Los estados ocultos del HMM coinciden en patrón con las actividades reales?
2. ¿Qué tan buenos son los HMM para detectar comportamientos sin etiquetas previas?
3. ¿Qué ajustes podrías hacer para mejorar el rendimiento del modelo?

## 📎 Conclusión

✅ Usaste un modelo oculto de Markov real sobre un dataset secuencial

✅ Entendiste cómo funciona el entrenamiento y la predicción de secuencias

✅ Visualizaste patrones ocultos e hiciste comparaciones reales

🔐 *“Los datos no siempre muestran todo... los HMM nos ayudan a ver lo que está escondido.”*