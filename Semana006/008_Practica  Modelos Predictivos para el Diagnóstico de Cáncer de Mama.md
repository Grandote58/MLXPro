# **Practica : Modelos Predictivos para el Diagnóstico de Cáncer de Mama**

**Proyecto:** *Construcción y Evaluación Crítica de un Sistema de Ayuda al Diagnóstico (CAD) para Cáncer de Mama*

Hoy nos adentramos en uno de los campos más impactantes del Machine Learning: la salud. Nuestra misión es construir un modelo capaz de clasificar tumores mamarios como benignos o malignos basándose en características numéricas extraídas de imágenes digitalizadas.

Cada decisión que tomemos, especialmente en cómo medimos el "éxito" de nuestro modelo, tiene un peso moral y ético. Nos enfocaremos obsesivamente en la pregunta: **"¿Qué significa un error en este contexto y cómo podemos minimizar el error más peligroso?"**

Abriremos Google Colab y trabajaremos con el famoso dataset "Breast Cancer Wisconsin", un recurso open-source fundamental para la enseñanza en este dominio.

# **Guía Paso a Paso**

## **Paso 0: Configuración del Entorno y Carga de Datos**

Iniciamos preparando nuestro laboratorio digital. Scikit-learn nos facilita el acceso a este dataset clásico, garantizando que todos trabajemos con los mismos datos de referencia.

```python
# Paso 0: Importar las librerías esenciales
# ------------------------------------------
# Pandas para la manipulación de datos y NumPy para operaciones numéricas
import pandas as pd
import numpy as np

# Matplotlib y Seaborn para visualizaciones de alta calidad
import matplotlib.pyplot as plt
import seaborn as sns

# Herramientas de Scikit-learn para el modelado
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             mean_squared_error, mean_absolute_error, r2_score)

# Configuración de estilo para los gráficos
%matplotlib inline
sns.set_style('darkgrid')
plt.style.use('seaborn-v0_8-deep')

# Cargar el dataset directamente desde scikit-learn
# ----------------------------------------------------
cancer_data = load_breast_cancer()
# Convertirlo a un DataFrame de Pandas para facilitar su uso
df = pd.DataFrame(data=cancer_data.data, columns=cancer_data.feature_names)
df['target'] = cancer_data.target # 0: Maligno, 1: Benigno

print("¡Dataset de Cáncer de Mama cargado con éxito!")
print(f"El dataset tiene {df.shape[0]} muestras y {df.shape[1]} columnas.")
df.head()
```

**Análisis del Código:**

- Usamos load_breast_cancer para importar el dataset. Es un método limpio y reproducible.
- Creamos un DataFrame de pandas, que es la estructura de datos más cómoda para el análisis.
- La columna target contiene nuestro objetivo: 0 para tumores malignos (el caso que queremos detectar) y 1 para benignos.

# **Parte 1: El Problema Crítico de Clasificación (Maligno vs. Benigno)**

### **Paso 1: Análisis Exploratorio de Datos (EDA)**

Antes de modelar, debemos entender la naturaleza de nuestros datos y, lo más importante, de nuestro objetivo.

```python
# Revisar la información general del dataset
print("\nInformación del DataFrame:")
df.info()

# Revisar el balance de clases. ¿Cuántos casos de cada tipo tenemos?
print("\nDistribución de Clases (Target):")
print(df['target'].value_counts())
# 0 = Maligno, 1 = Benigno

# Visualizar la distribución de clases
plt.figure(figsize=(8, 6))
sns.countplot(x='target', data=df, palette='viridis')
plt.title('Distribución de Clases (0: Maligno, 1: Benigno)')
plt.xticks([0, 1], ['Maligno', 'Benigno'])
plt.show()

# Veamos la correlación entre algunas características importantes
# Seleccionamos las primeras 10 características "mean" para no saturar el gráfico
plt.figure(figsize=(12, 10))
corr_matrix = df.iloc[:, :10].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de Correlación de Características "Mean"')
plt.show()
```

**Observaciones Clave:**

1. No hay valores nulos. El dataset está limpio.
2. Tenemos 357 tumores benignos (1) y 212 malignos (0). Es un dataset algo desbalanceado, pero no de forma extrema.
3. El mapa de calor muestra que muchas características están altamente correlacionadas (ej. mean_radius, mean_perimeter, mean_area). Esto es lógico y nos sugiere que la **regularización** podría ser útil para que el modelo no dependa demasiado de un solo grupo de características redundantes.

## **Paso 2: Preprocesamiento (División y Escalado)**

Este paso es fundamental para asegurar que nuestra evaluación sea honesta y que el modelo funcione correctamente.

```python
# Separar características (X) y objetivo (y)
X = df.drop('target', axis=1)
y = df['target']

# Training and test datasets
# ---------------------------
# Dividir los datos: 70% para entrenamiento, 30% para prueba.
# `stratify=y` es CRUCIAL aquí. Asegura que la proporción de casos malignos/benignos
# sea la misma en el conjunto de entrenamiento y en el de prueba.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Feature Scaling
# ---------------
# Los valores de las características tienen rangos muy diferentes. El escalado es necesario.
scaler = StandardScaler()

# Ajustamos el scaler SÓLO con los datos de entrenamiento (para evitar fuga de información)
X_train_scaled = scaler.fit_transform(X_train)
# Y luego aplicamos la misma transformación a los datos de prueba
X_test_scaled = scaler.transform(X_test)
```

**Análisis Pedagógico:**

- **stratify=y**: Imagina que, por mala suerte, casi todos los casos de cáncer (malignos) caen en el set de entrenamiento y casi ninguno en el de prueba. Nuestro test sería inútil. stratify previene esto, garantizando una división representativa.
- **StandardScaler**: Este proceso estandariza cada característica para que tenga una media de 0 y una desviación estándar de 1. Es vital para algoritmos como la Regresión Logística y las SVM, que son sensibles a la escala de los datos.

## **Paso 3: Entrenamiento y Evaluación Crítica del Modelo de Clasificación**

Entrenamos un modelo de Regresión Logística y nos sumergimos en el análisis de las métricas. **Aquí es donde separamos al técnico del científico de datos responsable.**

```python
# 1. Entrenar un modelo de Regresión Logística con Regularización
# ---------------------------------------------------------------
# Por defecto, LogisticRegression usa regularización L2 (Ridge).
# El parámetro C es la inversa de la fuerza de regularización.
# Un C más bajo significa una regularización más fuerte.
log_reg = LogisticRegression(C=1.0, random_state=42)
log_reg.fit(X_train_scaled, y_train)

# 2. Realizar predicciones en el conjunto de prueba
y_pred = log_reg.predict(X_test_scaled)

# 3. Evaluar el modelo: El momento de la verdad
print("\n--- Evaluación del Modelo de Clasificación ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precisión: {precision_score(y_test, y_pred, pos_label=0):.3f}") # Queremos la precisión para la clase 'Maligno'
print(f"Recall (Sensibilidad): {recall_score(y_test, y_pred, pos_label=0):.3f}") # Recall para 'Maligno'
print(f"F1-Score: {f1_score(y_test, y_pred, pos_label=0):.3f}") # F1 para 'Maligno'

# 4. Visualizar la Matriz de Confusión: La fuente de toda verdad
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Maligno (0)', 'Benigno (1)'])

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='inferno')
plt.title("Matriz de Confusión")
plt.show()
```

**Análisis de Métricas: La Decisión Clínica**

La clase positiva (la que queremos detectar) es 0 (Maligno). pos_label=0 en las métricas es fundamental.

- **Matriz de Confusión:**
  - **Verdaderos Positivos (TP, arriba-izquierda): 61** casos malignos que identificamos correctamente. ¡Éxito!
  - **Falsos Negativos (FN, arriba-derecha): 3** casos malignos que nuestro modelo clasificó como benignos. **ESTE ES EL ERROR MÁS PELIGROSO.** Significa que 3 pacientes con cáncer podrían ser enviados a casa sin tratamiento.
  - **Falsos Positivos (FP, abajo-izquierda): 1** caso benigno que etiquetamos como maligno. Esto causa ansiedad y probablemente una biopsia innecesaria, pero no es mortal.
  - **Verdaderos Negativos (TN, abajo-derecha): 106** casos benignos que clasificamos correctamente.
- **Recall (Sensibilidad): 0.953**. **Esta es la métrica REINA en este problema.**
  - **Pregunta que responde:** De todos los pacientes que *realmente tienen cáncer* en nuestro set de prueba (61+3=64), ¿a qué porcentaje detectamos?
  - **Interpretación:** Detectamos al 95.3% de los tumores malignos. Nuestro modelo **falló en detectar al 4.7%**. En un contexto clínico, el objetivo es llevar este número lo más cerca posible de 1.0 (100%), incluso si eso significa tener más falsos positivos.
- **Precisión: 0.984**.
  - **Pregunta que responde:** De todos los pacientes a los que les dijimos *que podrían tener cáncer* (61+1=62), ¿qué porcentaje realmente lo tenía?
  - **Interpretación:** El 98.4% de nuestras "alarmas" fueron correctas. Es una métrica muy alta, lo cual es bueno para no causar pánico innecesario.
- **F1-Score: 0.968**. Un balance armónico entre Precisión y Recall. Es una excelente métrica general, pero en este caso, debemos vigilar el Recall con más atención.
- **Accuracy: 0.977**. Un 97.7% de acierto global. Parece fantástico, pero oculta el hecho crítico de que fallamos en 3 casos de cáncer. **Nunca confíes solo en la Accuracy en problemas médicos.**

## **Paso 4: Validación Cruzada para una Evaluación Robusta**

¿Tuvimos suerte con nuestra división train_test_split? 

La validación cruzada (Cross-Validation) nos da una respuesta más fiable.

```python
# Usaremos 5 folds. El modelo se entrena y evalúa 5 veces.
# Puntuaremos usando 'recall' para la clase maligna, ya que es nuestra métrica clave.
cv_recall_scores = cross_val_score(log_reg, X_train_scaled, y_train, cv=5, scoring=lambda e, x, y: recall_score(y, e.predict(x), pos_label=0))

print("\n--- Validación Cruzada (Clasificación) ---")
print(f"Puntuaciones de Recall en cada fold: {cv_recall_scores}")
print(f"Recall Medio con CV: {cv_recall_scores.mean():.3f} (+/- {cv_recall_scores.std():.3f})")
```

**Análisis de CV:** Un Recall medio de 0.940 con una desviación estándar de 0.026 nos dice que el rendimiento de nuestro modelo es consistentemente alto y estable, no un golpe de suerte.

# **Parte 2: El Problema de Regresión (Creando un "Índice de Malignidad")**

Para cumplir con todos los objetivos de aprendizaje, vamos a crear un problema de regresión artificial pero didáctico. Nuestro objetivo será predecir un "Índice de Malignidad" continuo, en lugar de una clasificación binaria.

## **Paso 5: Ingeniería de Características - Creación del Objetivo de Regresión**

Crearemos un índice basado en las características más asociadas con la malignidad (valores más grandes suelen ser peores).

```python
# Creamos un "Índice de Malignidad" como una suma ponderada de características clave.
# Esto es un ejemplo de ingeniería de características para crear un proxy.
features_for_index = ['mean concave points', 'mean area', 'mean perimeter', 'mean texture']
df['malignancy_index'] = df[features_for_index].sum(axis=1)

# Visualicemos cómo se relaciona este nuevo índice con el diagnóstico real.
plt.figure(figsize=(10, 7))
sns.boxplot(x='target', y='malignancy_index', data=df, palette='magma')
plt.title('Índice de Malignidad vs. Diagnóstico Real')
plt.xticks([0, 1], ['Maligno', 'Benigno'])
plt.show()

# Preparar los datos para el nuevo problema de regresión
X_reg = df.drop(['target', 'malignancy_index'], axis=1) # Usamos las features originales
y_reg = df['malignancy_index']

# Nuevo split y escalado para el problema de regresión
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)
scaler_r = StandardScaler()
X_train_r_scaled = scaler_r.fit_transform(X_train_r)
X_test_r_scaled = scaler_r.transform(X_test_r)
```

**Análisis Pedagógico:** El boxplot muestra claramente que los tumores malignos (0) tienen un "Índice de Malignidad" mucho más alto y con más variabilidad. Nuestro índice sintético parece tener sentido y será un buen objetivo para un modelo de regresión.

## **Paso 6: Entrenamiento y Evaluación del Modelo de Regresión**

Usaremos Ridge, una regresión lineal con regularización L2, ideal para cuando las características están correlacionadas.

```python
# 1. Entrenar un modelo de Regresión Ridge
ridge_reg = Ridge(alpha=1.0) # alpha controla la fuerza de la regularización
ridge_reg.fit(X_train_r_scaled, y_train_r)

# 2. Realizar predicciones
y_pred_r = ridge_reg.predict(X_test_r_scaled)

# 3. Evaluar el modelo de regresión
mse = mean_squared_error(y_test_r, y_pred_r)
mae = mean_absolute_error(y_test_r, y_pred_r)
r2 = r2_score(y_test_r, y_pred_r)

print("\n--- Evaluación del Modelo de Regresión ---")
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"Error Absoluto Medio (MAE): {mae:.2f}")
print(f"Coeficiente de Determinación (R²): {r2:.3f}")

# 4. Visualizar los resultados
plt.figure(figsize=(10, 6))
plt.scatter(y_test_r, y_pred_r, alpha=0.6, c=y_pred_r, cmap='viridis')
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--', lw=2, label='Predicción Perfecta')
plt.xlabel("Índice de Malignidad Real")
plt.ylabel("Índice de Malignidad Predicho")
plt.title("Regresión Ridge: Real vs. Predicho")
plt.legend()
plt.colorbar(label='Valor Predicho')
plt.show()
```

**Análisis de Métricas (Regresión):**

- **R² (0.993):** Nuestro modelo explica el 99.3% de la varianza en el "Índice de Malignidad". Es un modelo extremadamente preciso. Esto es esperable, ya que el índice fue creado a partir de las mismas características que usamos para predecirlo.
- **MAE (21.60):** En promedio, nuestra predicción del índice se desvía en 21.60 puntos. Dado que los valores del índice van de ~150 a ~3300, este error es relativamente pequeño. Es la métrica más fácil de interpretar sobre la magnitud del error.
- **MSE (1115.89):** El error cuadrático penaliza más los errores grandes. Su raíz cuadrada (RMSE) sería ~33.4, que está en la misma escala que el MAE.

# **Tips para Practicar**

Hemos realizado un viaje completo, desde la clasificación de vida o muerte hasta la predicción de un índice de riesgo.

**Lecciones Clave:**

1. **El Contexto lo es Todo:** La elección de la métrica (Recall sobre Precisión en clasificación) no es una decisión técnica, sino una decisión ética y de negocio.
2. **Visualizar para Comprender:** Las matrices de confusión y los gráficos de dispersión no son adornos; son herramientas de diagnóstico para entender dónde y cómo falla nuestro modelo.
3. **La Robustez es la Meta:** La validación cruzada nos protege de celebrar victorias basadas en la suerte.

**Tips de Mejora:**

1. **Ajustar el Umbral de Decisión:** Por defecto, predict usa un umbral de probabilidad de 0.5. Usa log_reg.predict_proba(X_test_scaled) para obtener las probabilidades. Luego, puedes crear tu propia predicción cambiando el umbral (ej., y_pred_nuevo = (probabilidades[:, 0] > 0.1).astype(int)). Al bajar el umbral para la clase maligna, ¿qué pasa con el Recall y la Precisión? **¡Esta es una de las técnicas más poderosas para optimizar un modelo para un objetivo específico!**
2. **Probar Modelos Más Potentes:** Reemplaza LogisticRegression con SVC (Support Vector Classifier) o RandomForestClassifier. ¿Puedes reducir esos 3 Falsos Negativos a 2, 1 o incluso 0?
3. **Análisis de Características Importantes:** Entrena un RandomForestClassifier y usa model.feature_importances_ para ver qué características el modelo considera más importantes. ¿Coincide con la intuición médica?
4. **Pipelines para Profesionalizar:** Usa la clase Pipeline de scikit-learn para encadenar el escalador (StandardScaler) y el modelo (LogisticRegression) en un solo objeto. Esto hace que tu código sea más limpio, seguro y fácil de desplegar.