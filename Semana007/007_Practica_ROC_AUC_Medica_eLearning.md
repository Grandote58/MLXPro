
# **🧪 Práctica : Curva ROC y AUC con Datos Médicos**

👩‍⚕️ **Área:** Medicina

📊 **Tema:** Evaluación de Modelos con Curva ROC y AUC

🎯 **Objetivo:** Comprender y visualizar la **Curva ROC** y calcular el **AUC** como métrica para evaluar clasificadores en problemas médicos reales.

## 🔗 Paso 1: Cargar los datos médicos

Utilizaremos un dataset clínico sobre enfermedades cardíacas. La columna `target` indica si el paciente presenta enfermedad (`1`) o no (`0`).

```python
import pandas as pd

url = 'https://raw.githubusercontent.com/ageron/data/main/heart_disease/heart.csv'
df = pd.read_csv(url)
df.head()
```

## 🧠 Paso 2: Entender el problema

```python
df['target'].value_counts()
```

Este es un caso de **clasificación binaria** donde queremos medir qué tan bien el modelo puede distinguir entre pacientes sanos y enfermos.

---

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

## 📈 Paso 5: Calcular probabilidades para la Curva ROC

```python
# Probabilidades de clase positiva
y_scores = model.predict_proba(X_test)[:, 1]
```

## 🔍 Paso 6: Generar la Curva ROC y calcular AUC

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_scores)
auc = roc_auc_score(y_test, y_scores)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'Curva ROC (AUC = {auc:.2f})', color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('📉 Curva ROC - Evaluación de Modelo Médico')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
```

## 📌 Paso 7: Interpretación y Reflexión

### 🟢 Ventajas:
- Mide el rendimiento en todos los umbrales de decisión.
- El **AUC** resume la capacidad del modelo para discriminar entre clases.

### 🔴 Desventajas:
- No indica el punto de corte óptimo.
- Puede ser menos útil en casos con fuerte desbalance de clases.

## 🧠 Ejemplo Crítico

> En un diagnóstico clínico, la Curva ROC nos permite comparar modelos antes de definir un umbral de decisión,  
> como por ejemplo si queremos ser más estrictos para detectar una enfermedad sin aumentar mucho los falsos positivos.

## 🧩 Conclusión

🎯 La **Curva ROC** permite ver el comportamiento completo del modelo clasificando positivos y negativos.

📈 El **AUC** da una visión global: cuanto más cerca a 1, mejor el desempeño. Ideal para comparar múltiples modelos en medicina.

