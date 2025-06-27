
# **🧪 Práctica : F1 Score con Datos Médicos**

👩‍⚕️ **Área:** Medicina

📊 **Tema:** Métricas de Clasificación en Machine Learning

🎯 **Objetivo:** Comprender, calcular e interpretar la métrica **F1 Score** usando un conjunto de datos clínicos reales.

## 🔗 Paso 1: Cargar los datos médicos

Trabajaremos con un dataset de enfermedades cardíacas. La columna `target` indica si un paciente tiene (1) o no tiene (0) la enfermedad.

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

Este es un problema de **clasificación binaria**. El modelo debe predecir si un paciente está enfermo o no.

## ✂️ Paso 3: Dividir los datos

```python
from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```

## 🤖 Paso 4: Entrenar un modelo

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

## ✅ Paso 5: Calcular el F1 Score

El **F1 Score** es el promedio armónico entre precisión y recall. Es útil cuando hay desequilibrio en las clases.

```python
from sklearn.metrics import f1_score

y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred)

print(f"🎯 F1 Score del modelo: {f1:.2%}")
```

## 📈 Paso 6: Visualización con matriz de confusión

```python
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="BuPu")
plt.title("📊 Matriz de Confusión - F1 Score")
plt.show()
```

### Fórmulas útiles:

$$
\text{Precisión} = \frac{VP}{VP + FP}
\quad
\text{Recall} = \frac{VP}{VP + FN}
\quad
\text{F1} = 2 \cdot \frac{\text{Precisión} \cdot \text{Recall}}{\text{Precisión} + \text{Recall}}
$$

## 📌 Paso 7: Interpretación y Reflexión

### 🟢 Ventajas:
- Equilibra precisión y recall.
- Útil en **contextos con clases desbalanceadas**.

### 🔴 Desventajas:
- Menos intuitivo que otras métricas.
- Puede no reflejar bien los verdaderos negativos.

## 🧠 Ejemplo Crítico

> En diagnóstico de enfermedades, un modelo con buena precisión puede ser confiable al detectar enfermos,  
> y uno con buen recall detecta la mayoría de casos.  
> Pero **el F1 Score busca equilibrio** entre ambos aspectos para una visión global.

## 🧩 Conclusión

🎯 El **F1 Score** es una métrica robusta cuando necesitamos considerar tanto **la cobertura (recall)** como **la confianza en las predicciones positivas (precisión)**.

📘 Es ideal en el área médica cuando los errores de ambos tipos (falsos positivos y falsos negativos) tienen implicaciones importantes.

