# 🧬 **Práctica: “HMMGene: Detectando regiones codificantes en ADN con HMM 🧫🔬”**

### 🎯 **Objetivo**

Aplicar un **Modelo Oculto de Markov (HMM)** para **identificar regiones codificantes (genes)** y no codificantes en una secuencia real de ADN, utilizando observaciones de nucleótidos y estados ocultos como "gen" o "intergénico".

## 📂 **Dataset real utilizado**

- 🧠 **Nombre:** *Human DNA - Exon/Intron Data*
- 🌐 **URL Kaggle (descarga directa):**
   👉 https://www.kaggle.com/datasets/rodolfomendes/exon-intron-dna-sequences
- 📁 Archivo: `Human_DNA.csv`
- 🧬 Contiene secuencias de ADN anotadas como `exon` o `intron`, para modelar con dos estados ocultos.

## 👨‍🏫 **Paso a paso en Google Colab (detallado con emojis)**

### 🔹 1. Instalar librerías necesarias

```python
# ⚙️ Instalar hmmlearn si aún no está instalada
!pip install hmmlearn --quiet
```

### 🔹 2. Importar librerías

```python
# 📚 Librerías necesarias
import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
```

### 🔹 3. Cargar el dataset

🔽 **Sube `Human_DNA.csv` desde Kaggle a Google Colab**

```python
# 📂 Cargar archivo
df = pd.read_csv("Human_DNA.csv")
df = df.dropna()
df.head()
```

📌 El archivo contiene columnas como:

- `sequence`: cadena de nucleótidos (ej: AGCT...)
- `region`: tipo (exon/intron)

### 🔹 4. Preprocesamiento de secuencias

```python
# 🔬 Convertimos la secuencia en nucleótidos individuales
sequence_str = "".join(df['sequence'].values)
region_labels = df['region'].tolist()

# ✂️ Truncamos para que coincidan longitudes (tamaño educativo)
sequence_str = sequence_str[:1000]
region_labels = region_labels[:1000]

# 🔠 Codificamos los nucleótidos: A, C, G, T → 0, 1, 2, 3
nuc_encoder = LabelEncoder()
X = nuc_encoder.fit_transform(list(sequence_str)).reshape(-1, 1)
nuc_encoder.classes_
```

### 🔹 5. Configurar y entrenar el modelo HMM

```python
# 🧬 Entrenamos un modelo con 2 estados ocultos: exon (E) y intron (I)
model = hmm.MultinomialHMM(n_components=2, n_iter=100, random_state=42)
model.fit(X)

print("✅ Modelo entrenado con éxito")
```

### 🔹 6. Predecir estados ocultos

```python
# 🔍 Predecimos los estados ocultos de la secuencia de ADN
predicted_states = model.predict(X)

# 🧱 Mostrar los primeros 50 estados
print("🔐 Estados predichos (0: intrón, 1: exón):", predicted_states[:50])
```

### 🔹 7. Visualizar la secuencia de estados

```python
# 🎨 Gráfico de estados ocultos
plt.figure(figsize=(15,3))
plt.plot(predicted_states, color='green')
plt.title("🧬 Secuencia de estados ocultos predichos por el HMM")
plt.xlabel("Posición en la secuencia")
plt.ylabel("Estado oculto (0 = intron, 1 = exon)")
plt.grid(True)
plt.show()
```

## 🧠 Reflexión

1. ¿Cuántas veces cambia de estado el modelo? ¿Coincide con lo esperado?
2. ¿Qué patrones parecen tener los estados ocultos respecto a la secuencia de nucleótidos?
3. ¿Qué otras fuentes de datos podrían hacer más preciso el modelo?

## 📎 Conclusión

✅ Aplicaste HMM sobre una secuencia real de ADN

✅ Codificaste datos biológicos en variables modelables

✅ Detectaste patrones de probabilidad en regiones génicas

✅ Interpretaste visualmente el resultado del modelo

🧬 *“El ADN es el texto más antiguo. Los HMM nos ayudan a leer entre líneas.”*