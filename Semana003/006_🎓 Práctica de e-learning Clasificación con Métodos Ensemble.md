# **🎓 Práctica de e-learning: Clasificación con Métodos Ensemble**

## 🎯 Objetivos

- Comprender cómo funcionan los métodos ensemble como **Random Forest** y **Gradient Boosting**.
- Implementar modelos ensemble con datos reales.
- Comparar desempeño con modelos individuales.
- Visualizar la importancia de características y resultados.

## 📥 Paso 1: Importar librerías necesarias

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
```

## 📚 Paso 2: Cargar datos desde repositorio abierto

Usaremos el dataset **"Heart.csv"** desde GitHub para predecir enfermedades cardíacas.

🔗 URL de carga:
 https://raw.githubusercontent.com/ybifoundation/Dataset/main/Heart.csv

```python
url = "https://raw.githubusercontent.com/ybifoundation/Dataset/main/Heart.csv"
df = pd.read_csv(url)
df.head()
```

## 🧹 Paso 3: Preparación de los datos

```python
# Variables independientes y dependiente
X = df.drop(columns='HeartDisease')
y = df['HeartDisease']

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

## 🌳 Paso 4: Random Forest Classifier

```python
# Modelo Ensemble - Bagging
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluación
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Classifier")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
```

## ⚡ Paso 5: Gradient Boosting Classifier

```python
# Modelo Ensemble - Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# Evaluación
y_pred_gb = gb_model.predict(X_test)
print("Gradient Boosting Classifier")
print("Accuracy:", accuracy_score(y_test, y_pred_gb))
print(classification_report(y_test, y_pred_gb))
```

## 📊 Paso 6: Visualizar importancia de características

```python
# Importancia de características en Random Forest
importances = rf_model.feature_importances_
features = X.columns

plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=features)
plt.title("Importancia de Características - Random Forest")
plt.xlabel("Importancia")
plt.ylabel("Características")
plt.grid(True)
plt.show()
```

## ✅ Conclusiones

- Random Forest (bagging) y Gradient Boosting (boosting) combinan múltiples árboles para mejorar la predicción.
- Ambos métodos ofrecen **alta precisión** y permiten analizar qué variables influyen más.
- Son útiles en problemas reales donde un solo modelo puede fallar o sobreajustarse.