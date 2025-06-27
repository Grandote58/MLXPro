
# **📊 Práctica : Exactitud (Accuracy) con Datos Médicos**

👩‍⚕️ **Área:** Medicina | 🤖 **Tema:** Métricas de Evaluación en Machine Learning  

📘 **Objetivo:** Comprender, calcular y visualizar la métrica de exactitud usando datos reales.

## 🔗 Paso 1: Cargar los datos médicos

Usaremos un conjunto de datos de enfermedades cardíacas públicas. Contiene información clínica de pacientes, incluyendo si presentan enfermedad cardíaca (`target = 1`) o no (`target = 0`).

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

El problema es de **clasificación binaria**:  
- `0` → No tiene enfermedad cardíaca  
- `1` → Tiene enfermedad cardíaca  

El objetivo es entrenar un modelo que aprenda a predecir esta variable.

## ✂️ Paso 3: Dividir los datos

```python
from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```

## 🤖 Paso 4: Entrenar el modelo

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

## ✅ Paso 5: Calcular la Exactitud

```python
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Exactitud del modelo: {accuracy:.2%}")
```

## 📈 Paso 6: Visualización con matriz de confusión

La exactitud se basa en esta tabla visual:

```python
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues")
plt.title("🔍 Matriz de Confusión - Clasificación Binaria")
plt.show()
```

## 📌 Paso 7: Reflexión

### 🟢 Ventajas:
- Fácil de interpretar.
- Útil cuando las clases están balanceadas.

### 🔴 Desventajas:
- No distingue entre tipos de error.
- Puede ser engañosa si hay desbalance de clases.

## 💡 Ejemplo Crítico

> Si 90% de los pacientes están sanos y el modelo predice siempre "sano", tendrá 90% de exactitud, ¡pero no detectará ninguna enfermedad!

## 🧩 Conclusión

🔎 La exactitud es útil como primera métrica, pero debe complementarse con **precisión**, **recall** y **F1-score**, especialmente en contextos médicos donde los errores pueden ser costosos.

