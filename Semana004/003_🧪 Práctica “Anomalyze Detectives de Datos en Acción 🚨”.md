# 🧪 **Práctica: “Anomalyze: Detectives de Datos en Acción 🚨”**

### 🎯 **Objetivo de aprendizaje**

Aplicar el algoritmo **Isolation Forest** para detectar posibles anomalías (fraudes) en transacciones financieras simuladas. Se guiará paso a paso el análisis, el preprocesamiento, la detección de anomalías y su visualización.

### 🛠️ **Herramientas necesarias**

- Google Colab
- Librerías: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`

### 👨‍🏫 **PASO 1: Cargar librerías**

```python
# 📚 Librerías para manipulación de datos y visualización
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
```

### 📦 **PASO 2: Cargar el dataset simulado**

```python
# 📥 Simulamos un pequeño dataset con algunas anomalías conocidas
data = {
    'Time': [0, 10000, 20000, 30000, 40000, 50000],
    'V1': [-1.359807, 1.191857, -1.358354, -0.966272, 1.229657, -0.995920],
    'V2': [-0.072781, 0.266151, 1.340163, -0.185226, 0.141004, -0.218846],
    'V3': [2.536347, 0.166480, -1.119670, 1.792993, 0.045370, 1.074598],
    'Amount': [149.62, 2.69, 378.66, 123.50, 69.99, 200.00],
    'Class': [0, 0, 1, 0, 0, 1]  # 1 indica fraude, 0 es normal
}

df = pd.DataFrame(data)
df.head()
```

### 🧼 **PASO 3: Escalar los datos**

```python
# ✨ Preparamos los datos para el modelo
features = ['V1', 'V2', 'V3', 'Amount']
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])
```

### 🌲 **PASO 4: Aplicar Isolation Forest**

```python
# 🧠 Crear el modelo de detección de anomalías
model = IsolationForest(n_estimators=100, contamination=0.33, random_state=42)
df['Anomaly'] = model.fit_predict(df_scaled)

# 💡 Nota:
# -1 = anomalía detectada
#  1 = transacción normal
```

### 🔍 **PASO 5: Interpretar resultados**

```python
# 🧐 Comparar predicción vs. clase real
df[['Amount', 'Class', 'Anomaly']]
```

### 🎨 **PASO 6: Visualizar anomalías**

```python
# 🎯 Colores por tipo de transacción
sns.scatterplot(data=df, x='Amount', y='V1', hue='Anomaly', palette={1:'blue', -1:'red'})
plt.title('Visualización de Anomalías 🔍')
plt.xlabel('Monto de Transacción')
plt.ylabel('V1')
plt.grid(True)
plt.show()
```

### 🧠 **Preguntas de reflexión**

1. ¿Qué tan bien detectó el modelo las anomalías reales?
2. ¿Qué podría mejorar si tuvieras más variables?
3. ¿Qué significan las observaciones marcadas como "-1"?

### 🧾 **Conclusión**

Isolation Forest 🧭 es un algoritmo poderoso y eficiente para identificar comportamientos raros en grandes volúmenes de datos. En esta práctica aprendiste a escalar, modelar y visualizar anomalías en transacciones simuladas.