
# **🧪 Métricas de Clasificación con Datos Médicos**

👩‍⚕️ *Área: Medicina* 

🤖 *Tema: Evaluación de Modelos de Clasificación*

### 🎓 *Formato: e-Learning para plataformas LMS / Typora / Google Colab*

## 🎯 Objetivo

Aplicar y comprender las principales métricas de evaluación en modelos de clasificación binaria:
- Matriz de Confusión
- Exactitud (Accuracy)
- Precisión (Precision)
- Exhaustividad (Recall)
- F1 Score
- Especificidad
- Curva ROC y AUC



## 🔗 Paso 1: Cargar los datos médicos

```python
import pandas as pd

url = 'https://raw.githubusercontent.com/ageron/data/main/heart_disease/heart.csv'
datos = pd.read_csv(url)
datos.head()
```

## ✂️ Paso 2: Preparar y dividir el conjunto de datos

```python
from sklearn.model_selection import train_test_split

X = datos.drop('target', axis=1)
y = datos['target']

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.25, random_state=42)
```

## 🤖 Paso 3: Entrenar un modelo

```python
from sklearn.linear_model import LogisticRegression

modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_entrenamiento, y_entrenamiento)
```

## 🧩 Paso 4: Matriz de Confusión

```python
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_pred = modelo.predict(X_prueba)
ConfusionMatrixDisplay.from_estimator(modelo, X_prueba, y_prueba, cmap='Blues')
plt.title('📊 Matriz de Confusión')
plt.grid(False)
plt.show()
```

## ✅ Paso 5: Exactitud (Accuracy)

```python
from sklearn.metrics import accuracy_score

exactitud = accuracy_score(y_prueba, y_pred)
print(f"🎯 Exactitud del modelo: {exactitud:.2%}")
```

## 🎯 Paso 6: Precisión (Precision)

```python
from sklearn.metrics import precision_score

precision = precision_score(y_prueba, y_pred)
print(f"🔍 Precisión del modelo: {precision:.2%}")
```

## 🔍 Paso 7: Exhaustividad (Recall)

```python
from sklearn.metrics import recall_score

recall = recall_score(y_prueba, y_pred)
print(f"📈 Exhaustividad del modelo: {recall:.2%}")
```

## ⚖️ Paso 8: F1 Score

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_prueba, y_pred)
print(f"🔗 F1 Score del modelo: {f1:.2%}")
```

## 🧪 Paso 9: Especificidad

```python
from sklearn.metrics import confusion_matrix

matriz = confusion_matrix(y_prueba, y_pred)
tn, fp, fn, tp = matriz.ravel()
especificidad = tn / (tn + fp)
print(f"🧬 Especificidad del modelo: {especificidad:.2%}")
```

## 📉 Paso 10: Curva ROC y AUC

```python
from sklearn.metrics import roc_curve, roc_auc_score

y_scores = modelo.predict_proba(X_prueba)[:, 1]
fpr, tpr, _ = roc_curve(y_prueba, y_scores)
auc = roc_auc_score(y_prueba, y_scores)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.2f})', color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('📉 Curva ROC - Evaluación del Modelo')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
```

## 🧠 Reflexión Final

Cada métrica entrega una **perspectiva distinta** sobre el comportamiento del modelo:

- **Exactitud**: bien cuando las clases están balanceadas.
- **Precisión**: importante si los falsos positivos son costosos.
- **Recall**: clave si los falsos negativos son inaceptables.
- **F1 Score**: balance entre precisión y recall.
- **Especificidad**: evita clasificar sanos como enfermos.
- **AUC ROC**: mide qué tan bien el modelo discrimina clases en todos los umbrales.

