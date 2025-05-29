## 🏥 Práctica: Predicción de Enfermedades Cardíacas con Aprendizaje Supervisado

### 🎯 Objetivo

Desarrollar un modelo de aprendizaje supervisado para predecir la presencia de enfermedades cardíacas en pacientes, utilizando características clínicas y demográficas.

------

### 📥 Paso 1: Importar Librerías Necesarias

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
```

### 📚 Paso 2: Cargar el Conjunto de Datos

Utilizaremos el conjunto de datos de enfermedades cardíacas disponible en el UCI Machine Learning Repository.

🔗 URL de descarga: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data

```python
# Definir nombres de columnas según la documentación
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
    'ca', 'thal', 'target'
]

# Cargar los datos
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
df = pd.read_csv(url, names=column_names)
```

### 🧹 Paso 3: Exploración y Limpieza de Datos

```python
# Reemplazar los signos de interrogación por NaN
df.replace('?', np.nan, inplace=True)

# Convertir columnas a tipo numérico
for col in ['ca', 'thal']:
    df[col] = pd.to_numeric(df[col])

# Eliminar filas con valores faltantes
df.dropna(inplace=True)

# Convertir la columna objetivo a binaria: 0 (sin enfermedad), 1 (con enfermedad)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# Mostrar información general
df.info()
```

### 📊 Paso 4: Análisis Exploratorio de Datos

```python
# Distribución de la variable objetivo
sns.countplot(x='target', data=df)
plt.title('Distribución de la Variable Objetivo')
plt.xlabel('Presencia de Enfermedad Cardíaca')
plt.ylabel('Cantidad')
plt.show()

# Correlación entre variables
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.title('Matriz de Correlación')
plt.show()
```

### ✂️ Paso 5: División de los Datos

```python
# Separar características y variable objetivo
X = df.drop('target', axis=1)
y = df['target']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 🤖 Paso 6: Selección y Entrenamiento de Modelos

#### Modelo 1: Regresión Logística

```python
# Crear y entrenar el modelo
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# Realizar predicciones
y_pred_log = log_model.predict(X_test)
```

#### Modelo 2: Bosque Aleatorio (Random Forest)

```python
# Crear y entrenar el modelo
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Realizar predicciones
y_pred_rf = rf_model.predict(X_test)
```

### 📈 Paso 7: Evaluación de Modelos

#### Regresión Logística

```python
print("Regresión Logística:")
print("Exactitud:", accuracy_score(y_test, y_pred_log))
print("Reporte de Clasificación:\n", classification_report(y_test, y_pred_log))
```

#### Bosque Aleatorio

```python
print("Bosque Aleatorio:")
print("Exactitud:", accuracy_score(y_test, y_pred_rf))
print("Reporte de Clasificación:\n", classification_report(y_test, y_pred_rf))
```

### 📊 Paso 8: Visualización de Resultados

#### a) Matriz de Confusión

```python
# Matriz de confusión para Bosque Aleatorio
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión - Bosque Aleatorio')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.show()
```

#### b) Importancia de Características

```python
# Importancia de características en Bosque Aleatorio
importances = rf_model.feature_importances_
features = X.columns
indices = np.argsort(importances)

plt.figure(figsize=(10,6))
plt.title('Importancia de Características')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Importancia Relativa')
plt.show()
```

### ✅ Conclusión

En esta práctica, hemos desarrollado y evaluado dos modelos de aprendizaje supervisado para predecir la presencia de enfermedades cardíacas. Ambos modelos han mostrado un desempeño razonable, siendo el Bosque Aleatorio ligeramente superior en términos de exactitud. 

La visualización de la importancia de características nos ha permitido identificar las variables más influyentes en la predicción, lo cual es valioso para la interpretación clínica y la toma de decisiones.