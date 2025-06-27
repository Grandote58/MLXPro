
# **🧪 Práctica : Precisión (Precision) con Datos Médicos**

👩‍⚕️ **Área:** Medicina | 📊 **Tema:** Métricas de Evaluación en Machine Learning 

🎯 **Objetivo:** Aprender qué es la *precisión*, cómo se calcula y cuándo utilizarla con un ejemplo real de clasificación médica.

## 🔗 Paso 1: Cargar los datos médicos

Usaremos un dataset sobre enfermedades cardíacas con información clínica de pacientes. La variable `target` nos indica si un paciente tiene o no enfermedad cardíaca.

```python
import pandas as pd

# Cargar datos
url = 'https://raw.githubusercontent.com/ageron/data/main/heart_disease/heart.csv'
df = pd.read_csv(url)
df.head()
```

## 🧠 Paso 2: Comprender el problema

```python
df['target'].value_counts()
```

La tarea es predecir si un paciente tiene enfermedad (`1`) o no (`0`).  
Es un problema de **clasificación binaria**.

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

## ✅ Paso 5: Calcular la Precisión

```python
from sklearn.metrics import precision_score

y_pred = model.predict(X_test)
precision = precision_score(y_test, y_pred)

print(f"🎯 Precisión del modelo: {precision:.2%}")
```



## 📈 Paso 6: Visualización con matriz de confusión

```python
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Purples')
plt.title("🔍 Matriz de Confusión - Precisión")
plt.show()
```

La precisión se calcula así:

Donde:
- **VP (Verdaderos Positivos):** Casos positivos correctamente clasificados.
- **FP (Falsos Positivos):** Casos negativos incorrectamente clasificados como positivos.

## 📌 Paso 7: Interpretación y Reflexión

### 🟢 Ventajas:
- Útil cuando **los falsos positivos son costosos**.
- Se enfoca en la confiabilidad de las predicciones positivas.

### 🔴 Desventajas:
- Ignora los falsos negativos.
- Puede ser alta aunque no se detecten todos los casos positivos.

## 🧠 Ejemplo Crítico

> En un sistema de detección de cáncer, una alta precisión implica que **casi todos los pacientes detectados como enfermos realmente lo están**.  
> Pero si el modelo no detecta a todos los enfermos, entonces la precisión puede ser buena mientras que el *recall* es bajo.

## 🧩 Conclusión

🎓 La **precisión** es clave cuando deseamos minimizar falsos positivos, por ejemplo en sistemas de diagnóstico automático donde las alertas erróneas pueden generar ansiedad o procedimientos innecesarios.

📌 Para una evaluación completa, conviene combinarla con *recall* y *F1-score*.

