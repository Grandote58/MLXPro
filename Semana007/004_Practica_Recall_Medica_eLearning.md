
# **🧪 Práctica : Exhaustividad (Recall) con Datos Médicos**

👩‍⚕️ **Área:** Medicina

📊 **Tema:** Evaluación de Modelos con Métricas de Clasificación

🎯 **Objetivo:** Comprender, calcular e interpretar la métrica de *recall* (exhaustividad) en un caso real usando datos clínicos.

## 🔗 Paso 1: Cargar los datos médicos

Usaremos un dataset sobre enfermedades cardíacas, donde la columna `target` indica si un paciente tiene (1) o no (0) una enfermedad cardíaca.

```python
import pandas as pd

url = 'https://raw.githubusercontent.com/ageron/data/main/heart_disease/heart.csv'
df = pd.read_csv(url)
df.head()
```

## 🧠 Paso 2: Comprender el problema

```python
df['target'].value_counts()
```

Este es un problema de **clasificación binaria**. Nuestro modelo debe predecir si un paciente está enfermo o sano.

## ✂️ Paso 3: Dividir los datos

```python
from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```

## 🤖 Paso 4: Entrenar un modelo de clasificación

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

## ✅ Paso 5: Calcular la métrica de Recall (Exhaustividad)

```python
from sklearn.metrics import recall_score

y_pred = model.predict(X_test)
recall = recall_score(y_test, y_pred)

print(f"🔍 Exhaustividad (Recall): {recall:.2%}")
```

## 📈 Paso 6: Visualización con matriz de confusión

```python
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Oranges')
plt.title("📊 Matriz de Confusión - Recall")
plt.show()
```

La fórmula para Recall es:
$$
\text{Recall} = \frac{TP}{TP + FN}
$$
nde:
- **TP (Verdaderos Positivos):** Enfermos correctamente identificados.
- **FN (Falsos Negativos):** Enfermos clasificados como sanos.

## 📌 Paso 7: Interpretación y Reflexión

### 🟢 Ventajas:
- Muy útil cuando **es crítico detectar todos los casos positivos**.
- Ideal en salud, seguridad y fraude.

### 🔴 Desventajas:
- Puede generar más falsos positivos.
- No mide qué tan confiables son los positivos.

## 🧠 Ejemplo Crítico

> En un test de cáncer, **recall alto** significa que casi todos los pacientes enfermos son detectados.  
> Esto es vital, ya que pasar por alto un caso positivo puede tener consecuencias graves.

## 🧩 Conclusión

🎯 La exhaustividad es crucial cuando el **riesgo de omitir un caso positivo es inaceptable**.  
Aunque puede reducir la precisión, es preferible **"detectar más aunque nos equivoquemos más"** en escenarios médicos.

