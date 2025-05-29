# **🌳 Práctica e-learning: Clasificación con Random Forest**

## 🎯 Objetivos

- Comprender el funcionamiento del algoritmo Random Forest.
- Aplicarlo sobre un conjunto de datos abierto y real.
- Evaluar su desempeño y visualizar la importancia de características.
- Interpretar resultados gráficamente.

## 🧰 Paso 1: Importar librerías necesarias

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

## 📥 Paso 2: Cargar el conjunto de datos

Usaremos el dataset de **enfermedades cardíacas** desde el repositorio de YBI Foundation en GitHub.

🔗 URL:
 https://raw.githubusercontent.com/ybifoundation/Dataset/main/Heart.csv

```python
# Cargar el dataset
url = "https://raw.githubusercontent.com/ybifoundation/Dataset/main/Heart.csv"
df = pd.read_csv(url)

# Ver primeras filas
df.head()
```

## ✂️ Paso 3: Preparación de los datos

```python
# Variables predictoras y objetivo
X = df.drop(columns='HeartDisease')
y = df['HeartDisease']

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

## 🌳 Paso 4: Entrenamiento del modelo Random Forest

```python
# Crear el modelo con 100 árboles
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```

## 📊 Paso 5: Evaluación del modelo

```python
# Realizar predicciones
y_pred = rf_model.predict(X_test)

# Evaluación
print("Exactitud (Accuracy):", accuracy_score(y_test, y_pred))
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))
```

## 📈 Paso 6: Visualización de resultados

### a) Matriz de confusión

```python
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.show()
```

### b) Importancia de características

```python
# Visualizar importancia de variables
importances = rf_model.feature_importances_
features = X.columns

plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=features)
plt.title("Importancia de Características - Random Forest")
plt.xlabel("Importancia Relativa")
plt.ylabel("Variables")
plt.grid(True)
plt.show()
```

## ✅ Conclusión

- **Random Forest** es un método de bagging que combina múltiples árboles de decisión para una mejor precisión.
- Tiene buena capacidad de generalización y permite identificar variables más influyentes.
- Ideal para clasificación binaria y multiclase, incluso con variables correlacionadas.