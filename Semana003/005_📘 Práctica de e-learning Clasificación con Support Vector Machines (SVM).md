# 📘 Práctica de e-learning: Clasificación con Support Vector Machines (SVM)

## 🎯 Objetivo:

- Comprender cómo funciona el algoritmo SVM.
- Aplicarlo a un conjunto de datos real y abierto.
- Visualizar el margen y los vectores de soporte para una mejor interpretación.

## 🧰 Paso 1: Preparar el entorno

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
```

## 📥 Paso 2: Cargar datos desde un repositorio open source

Usaremos el dataset de **clasificación de cáncer de mama** desde el repositorio UCI disponible en `sklearn.datasets`.

```python
from sklearn.datasets import load_breast_cancer

# Cargar dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Revisar estructura
X.head()
```

## ✂️ Paso 3: División de datos y escalado

```python
# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar características (muy importante para SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## 🧠 Paso 4: Entrenar modelo SVM

```python
# Crear y entrenar el modelo
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train_scaled, y_train)
```

## 📊 Paso 5: Evaluar el modelo

```python
# Predicciones
y_pred = svm_model.predict(X_test_scaled)

# Evaluación
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))
```

## 📈 Paso 6: Visualización 2D (usando PCA para reducir dimensiones)

```python
from sklearn.decomposition import PCA

# Reducimos a 2 dimensiones para visualización
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)
X_pca_test = pca.transform(X_test_scaled)

# Entrenar SVM con datos reducidos
svm_vis = SVC(kernel='linear', C=1.0)
svm_vis.fit(X_pca, y_train)

# Crear malla para graficar frontera
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

Z = svm_vis.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Gráfico
plt.figure(figsize=(10, 6))
plt.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors='black')
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='coolwarm', s=30, edgecolors='k')
plt.title("SVM: Separación lineal con PCA (2D)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.show()
```

## ✅ Conclusiones de Aprendizaje

- SVM maximiza el margen entre clases, logrando buena generalización.
- Es muy eficaz en espacios de alta dimensión y puede adaptarse a relaciones no lineales mediante kernels.
- La visualización con PCA ayuda a interpretar modelos en datasets complejos.