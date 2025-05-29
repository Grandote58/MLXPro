# **⚡ Práctica e-learning: Clasificación con Gradient Boosting**

## 🎯 Objetivos

- Comprender cómo funciona el algoritmo de boosting mediante `GradientBoostingClassifier`.
- Aplicarlo a un dataset real y de acceso libre.
- Evaluar el rendimiento del modelo.
- Visualizar e interpretar las variables más importantes.

## 🧰 Paso 1: Importar Librerías

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

## 📥 Paso 2: Cargar Dataset desde Repositorio Abierto

Usamos el dataset de enfermedades cardíacas, disponible públicamente en GitHub:

🔗 URL:
 https://raw.githubusercontent.com/ybifoundation/Dataset/main/Heart.csv

```python
# Cargar datos desde GitHub
url = "https://raw.githubusercontent.com/ybifoundation/Dataset/main/Heart.csv"
df = pd.read_csv(url)

# Mostrar las primeras filas
df.head()
```

## ✂️ Paso 3: Preparación de Datos

```python
# Separar características y variable objetivo
X = df.drop(columns='HeartDisease')
y = df['HeartDisease']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

## ⚙️ Paso 4: Entrenar el Modelo Gradient Boosting

```python
# Crear y ajustar el modelo
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)
```

## 📊 Paso 5: Evaluar el Modelo

```python
# Predicción
y_pred = gb_model.predict(X_test)

# Métricas
print("Exactitud:", accuracy_score(y_test, y_pred))
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))
```

## 📈 Paso 6: Visualizar Resultados

### a) Matriz de Confusión

```python
# Visualización de matriz
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens')
plt.title("Matriz de Confusión - Gradient Boosting")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.show()
```

### b) Importancia de Variables

```python
# Importancia de características
importances = gb_model.feature_importances_
features = X.columns

plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=features)
plt.title("Importancia de Características - Gradient Boosting")
plt.xlabel("Importancia Relativa")
plt.ylabel("Variables")
plt.grid(True)
plt.show()
```

## ✅ Conclusión

- Gradient Boosting mejora modelos secuencialmente, corrigiendo errores del modelo anterior.
- Es poderoso para tareas de clasificación compleja.
- La visualización ayuda a identificar qué variables más contribuyen al resultado.