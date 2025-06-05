# 🧪 **Práctica: “🔐 FRAUDE DETECTADO: Analizando Transacciones con ML”**

### 🎯 **Objetivo**

Aplicar el modelo **Isolation Forest** sobre un conjunto de datos real de transacciones con tarjeta de crédito, para detectar posibles fraudes utilizando aprendizaje no supervisado.

## 📂 Dataset real

- **Nombre:** Credit Card Fraud Detection
- **Fuente:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Archivo principal:** `creditcard.csv`
- 🔐 284,807 transacciones | 492 fraudes | Datos transformados con PCA

📥 **Nota**: Para usarlo en Google Colab, sube el archivo manualmente desde tu dispositivo o monta tu cuenta de Google Drive.

## 👨‍🏫 Paso a paso en Google Colab con explicación

### 1️⃣ Importar librerías necesarias

```python
# 📚 Librerías para análisis y visualización
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
```

### 2️⃣ Cargar el dataset real

```python
# 📂 Asegúrate de subir 'creditcard.csv' al entorno de Colab
df = pd.read_csv('creditcard.csv')
df.shape
```

🔎 El dataset tiene más de 284 mil registros. Cada columna `V1` a `V28` es una transformación PCA.

### 3️⃣ Vista rápida de los datos

```python
# 👀 Vista general
df.head()

# 🔍 Verificar balanceo de clases
print(df['Class'].value_counts())
sns.countplot(x='Class', data=df)
plt.title("Distribución: Transacciones legítimas vs fraudulentas 🚨")
plt.show()
```

📌 `Class = 1` representa fraude — extremadamente desbalanceado.

### 4️⃣ Escalar las variables

```python
# ⚙️ Seleccionar variables numéricas + 'Amount'
features = df.drop(columns=['Time', 'Class'])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
```

### 5️⃣ Aplicar Isolation Forest

```python
# 🌲 Entrenamos el modelo con 0.0018 de contaminación (~proporción de fraudes)
model = IsolationForest(n_estimators=100, contamination=0.0018, random_state=42)
df['Anomaly'] = model.fit_predict(scaled_features)

# 🧾 Interpretación:
# Anomaly = -1 → Anomalía (posible fraude)
# Anomaly = 1  → Normal
```

### 6️⃣ Comparar predicciones vs casos reales

```python
# 🔍 Ver matriz de conteo
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(df['Class'], df['Anomaly'] == -1)
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Real Normal', 'Real Fraude'], columns=['Detectado Normal', 'Detectado Fraude'])
conf_matrix_df
```

### 7️⃣ Visualización

```python
# 🎯 Visualizar con V1 y V2 como ejemplo
sns.scatterplot(data=df.sample(10000), x='V1', y='V2', hue='Anomaly', palette={1:'blue', -1:'red'})
plt.title("Visualización de anomalías con Isolation Forest 🔍")
plt.xlabel('V1')
plt.ylabel('V2')
plt.legend(title='Tipo')
plt.grid(True)
plt.show()
```

## 🧠 Reflexión

- ¿Qué proporción de fraudes reales logró detectar el modelo?
- ¿Cuál fue la tasa de falsos positivos?
- ¿Qué limitaciones encuentras en este modelo?
- ¿Cómo mejorarías este sistema con datos adicionales?

## 🧾 Conclusión

En esta práctica aprendiste a:

- Usar un dataset real y extenso
- Escalar características y preparar datos reales
- Aplicar *Isolation Forest* para detectar anomalías
- Visualizar e interpretar los resultados
- Evaluar la efectividad del modelo de forma crítica

🧠 *“Detectar lo raro es tan valioso como entender lo común. Los fraudes no se etiquetan... se descubren.”*

