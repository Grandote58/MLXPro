
# **🧪 Práctica : Especificidad con Datos Médicos**

👩‍⚕️ **Área:** Medicina

📊 **Tema:** Métricas de Evaluación de Modelos

🎯 **Objetivo:** Comprender, calcular e interpretar la **especificidad** (también llamada tasa de verdaderos negativos) usando datos reales del ámbito clínico.

## 🔗 Paso 1: Cargar los datos médicos

Usaremos un dataset de enfermedades cardíacas. La columna `target` nos indica si un paciente tiene (`1`) o no (`0`) enfermedad cardíaca.

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

Queremos predecir si un paciente **no tiene enfermedad cardíaca** (`0`).  
Por lo tanto, nos interesará la proporción de negativos verdaderos correctamente identificados: **especificidad**.

## ✂️ Paso 3: Dividir el conjunto de datos

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

## ✅ Paso 5: Calcular la Especificidad

La especificidad se calcula como:
$$
\text{Especificidad} = \frac{TN}{TN + FP}
$$
No está incluida directamente en sklearn, pero puede obtenerse desde la matriz de confusión:

```python
from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

specificity = tn / (tn + fp)
print(f"🧪 Especificidad del modelo: {specificity:.2%}")
```

## 📈 Paso 6: Visualización con matriz de confusión

```python
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Greens")
plt.title("📊 Matriz de Confusión - Especificidad")
plt.show()
```

## 📌 Paso 7: Interpretación y Reflexión

### 🟢 Ventajas:
- Útil cuando es crítico **evitar falsos positivos**.
- Mide la capacidad del modelo para **identificar correctamente a los pacientes sanos**.

### 🔴 Desventajas:
- No toma en cuenta los verdaderos positivos.
- Puede dar falsa seguridad si se prioriza más que el recall.

## 🧠 Ejemplo Crítico

> En un examen antidopaje o de enfermedades infecciosas, una alta **especificidad** asegura que las personas sanas no sean clasificadas erróneamente como enfermas (evita estigmatización o tratamientos innecesarios).

## 🧩 Conclusión

🔎 La **especificidad** se enfoca en **los verdaderos negativos**.  
Es indispensable en medicina cuando los **falsos positivos pueden causar ansiedad, tratamiento erróneo o costos innecesarios**.
