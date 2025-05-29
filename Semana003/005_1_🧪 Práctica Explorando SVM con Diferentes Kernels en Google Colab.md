# **🧪 Práctica: Explorando SVM con Diferentes Kernels en Google Colab**

## 🎯 Objetivos de Aprendizaje

- Comprender cómo las funciones kernel transforman los datos para permitir la separación no lineal.
- Implementar y comparar SVM con kernels lineal, polinomial y RBF.
- Visualizar las fronteras de decisión generadas por cada kernel.

## 📥 Paso 1: Importar Librerías Necesarias

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
```

## 📚 Paso 2: Cargar y Preparar el Conjunto de Datos

Utilizaremos el conjunto de datos **Iris**, disponible en Scikit-learn, y seleccionaremos dos clases para facilitar la visualización.

```python
# Cargar el conjunto de datos Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Seleccionar solo las clases 0 y 1 para clasificación binaria
X = X[y != 2]
y = y[y != 2]

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

## ⚙️ Paso 3: Entrenar Modelos SVM con Diferentes Kernels

Entrenaremos tres modelos SVM utilizando kernels lineal, polinomial y RBF.

```python
# SVM con kernel lineal
svc_linear = SVC(kernel='linear')
svc_linear.fit(X_train, y_train)

# SVM con kernel polinomial
svc_poly = SVC(kernel='poly', degree=3)
svc_poly.fit(X_train, y_train)

# SVM con kernel RBF
svc_rbf = SVC(kernel='rbf')
svc_rbf.fit(X_train, y_train)
```

## 📈 Paso 4: Visualizar las Fronteras de Decisión

Para visualizar las fronteras de decisión, reduciremos las características a dos dimensiones utilizando solo las dos primeras características del conjunto de datos.

```python
def plot_decision_boundary(model, X, y, title):
    # Seleccionar solo las dos primeras características
    X = X[:, :2]
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.02  # Paso de la malla

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.title(title)
    plt.show()

# Aplicar la función de visualización a cada modelo
plot_decision_boundary(svc_linear, X_train, y_train, 'SVM con Kernel Lineal')
plot_decision_boundary(svc_poly, X_train, y_train, 'SVM con Kernel Polinomial (grado 3)')
plot_decision_boundary(svc_rbf, X_train, y_train, 'SVM con Kernel RBF')
```

## ✅ Conclusiones

- El **kernel lineal** es adecuado cuando los datos son linealmente separables.
- El **kernel polinomial** permite capturar relaciones no lineales mediante la transformación de las características.
- El **kernel RBF** es eficaz para manejar datos con fronteras de decisión complejas y no lineales.