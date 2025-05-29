# **Práctica:** **"Diagnóstico Asistido por IA: Clasificando Tumores Mamarios con Python 🩺🔬"**

**Objetivo:** Construir y evaluar dos modelos de aprendizaje supervisado para clasificar tumores mamarios como benignos o malignos, basándose en características extraídas de imágenes digitalizadas de aspirados con aguja fina (FNA).

**Herramientas:** Visual Studio Code o Google Colab con Python.

**Dataset:** Breast Cancer Wisconsin (Diagnostic)

- **URL para descargar (CSV):** [https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data](https://www.google.com/url?sa=E&q=https%3A%2F%2Farchive.ics.uci.edu%2Fml%2Fmachine-learning-databases%2Fbreast-cancer-wisconsin%2Fwdbc.data)
- **Información del dataset (nombres de columnas, etc.):** [https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names](https://www.google.com/url?sa=E&q=https%3A%2F%2Farchive.ics.uci.edu%2Fml%2Fmachine-learning-databases%2Fbreast-cancer-wisconsin%2Fwdbc.names)

## **PASO A PASO DETALLADO 📝**

### **Paso 0: Configuración del Entorno y Librerías 🛠️**

```python
# Importación de librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

# Configuraciones para visualizaciones (opcional, pero recomendado)
plt.style.use('seaborn-v0_8-whitegrid') # Estilo de gráficos agradable
sns.set_palette('pastel') # Paleta de colores
print("Librerías importadas exitosamente! 🎉")
```

### **Paso 1: Carga y Exploración Inicial de los Datos 📊**

Vamos a cargar los datos desde la URL de UCI. El archivo wdbc.data no tiene encabezados, así que se los añadiremos según el archivo wdbc.names.

```python
# URL del dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

# Nombres de las columnas (según wdbc.names)
# La primera columna es ID (que descartaremos), la segunda es Diagnosis (M=malignant, B=benign)
# Las siguientes 30 son las características numéricas.
column_names = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(1, 31)]

# Cargar el dataset
df = pd.read_csv(url, header=None, names=column_names)

print("Dataset cargado! primeras 5 filas: ")
print(df.head())
print("\nInformación general del DataFrame:")
df.info()
print("\nEstadísticas descriptivas:")
print(df.describe())
print("\nConteo de valores nulos por columna:")
print(df.isnull().sum())
```

#### **Observaciones de la Exploración Inicial:**

- Vemos que la columna id no será útil para la predicción, así que la eliminaremos.
- La columna diagnosis es nuestro objetivo (target) y es categórica ('M' o 'B'). Necesitaremos convertirla a numérica.
- Las otras 30 columnas (feature_1 a feature_30) son las características predictoras. Todas parecen ser numéricas.
- No hay valores nulos, ¡lo cual es genial! 👍

### **Paso 2: Limpieza y Preprocesamiento de Datos 🧹✨**

1. **Eliminar columna ID:**
2. **Codificar la variable objetivo diagnosis:** 'M' (maligno) a 1 y 'B' (benigno) a 0.
3. **Separar características (X) y objetivo (y).**
4. **Escalar las características:** Muchos algoritmos (como Regresión Logística y SVM) funcionan mejor o son más estables cuando las características numéricas están en una escala similar. Usaremos StandardScaler.

```python
# 1. Eliminar columna ID
df = df.drop('id', axis=1)
print("\nColumna 'id' eliminada.")

# 2. Codificar la variable objetivo 'diagnosis'
# Usaremos LabelEncoder, pero también se puede hacer con map: df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])
# M (maligno) se mapea a 1, B (benigno) se mapea a 0. Guardamos las clases para referencia.
# print(f"Clases aprendidas por LabelEncoder: {le.classes_} -> {le.transform(le.classes_)}")
print(f"\nVariable 'diagnosis' codificada: (M={le.transform(['M'])[0]}, B={le.transform(['B'])[0]})")
print("Primeras filas con 'diagnosis' codificada:")
print(df.head())

# 3. Separar características (X) y objetivo (y)
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
print(f"\nDimensiones de X: {X.shape}")
print(f"Dimensiones de y: {y.shape}")

# 4. Escalar las características
# Es importante escalar DESPUÉS de dividir en train/test para evitar data leakage del test set al training set.
# Sin embargo, para la exploración inicial (como el heatmap de correlación) podemos usar X sin escalar o escalarlo completo temporalmente.
# Aquí, escalaremos después de la división.

print("\nPreprocesamiento básico completado! 🧼")
```

### **Paso 3: División de los Datos en Conjuntos de Entrenamiento y Prueba**

Dividiremos los datos para poder entrenar el modelo con una porción y evaluarlo con otra porción que no ha visto antes.

```python
# Dividir los datos: 70% para entrenamiento, 30% para prueba
# random_state asegura reproducibilidad. stratify=y asegura que la proporción de clases sea similar en train y test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Tamaño de X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"Tamaño de y_train: {y_train.shape}, y_test: {y_test.shape}")
print(f"Proporción de clases en y_train:\n{y_train.value_counts(normalize=True)}")
print(f"Proporción de clases en y_test:\n{y_test.value_counts(normalize=True)}")

# Ahora sí, escalamos las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Ajusta el scaler en train y lo transforma
X_test_scaled = scaler.transform(X_test)     # Solo transforma el test con el scaler ajustado en train

# Convertir de nuevo a DataFrames para mantener los nombres de las columnas (opcional, pero útil para inspección)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print("\nDatos divididos y escalados! Listos para el modelado. 🚂")
```

### **Paso 4: Selección, Entrenamiento y Evaluación del Modelo 1 - Regresión Logística 📈**

#### **¿Por qué Regresión Logística?**

- Es un modelo lineal simple, rápido de entrenar e interpretable.
- Proporciona probabilidades, lo cual puede ser útil.
- Es un buen modelo base para problemas de clasificación binaria.
- Funciona bien cuando las características están escaladas.

```python
print("\n--- Modelo 1: Regresión Logística ---")
# Instanciar el modelo
log_reg = LogisticRegression(random_state=42, solver='liblinear') # liblinear es bueno para datasets pequeños

# Entrenar el modelo
log_reg.fit(X_train_scaled, y_train)
print("Modelo de Regresión Logística entrenado. ✅")

# Realizar predicciones en el conjunto de prueba
y_pred_log_reg = log_reg.predict(X_test_scaled)
y_pred_proba_log_reg = log_reg.predict_proba(X_test_scaled)[:, 1] # Probabilidades para la clase positiva (1)

# Evaluar el modelo
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)
class_report_log_reg = classification_report(y_test, y_pred_log_reg)

print(f"\nAccuracy (Regresión Logística): {accuracy_log_reg:.4f}")
print(f"AUC-ROC (Regresión Logística): {roc_auc_score(y_test, y_pred_proba_log_reg):.4f}")
print("\nMatriz de Confusión (Regresión Logística):")
# Usamos un heatmap para la matriz de confusión para que sea más visual
# Clases: 0 (Benigno), 1 (Maligno)
# Mapeo original: M=1, B=0. le.classes_ nos da ['B', 'M'] que mapean a [0, 1]
class_names = le.classes_ # ['B', 'M']
sns.heatmap(conf_matrix_log_reg, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión - Regresión Logística')
plt.show()

print("\nInforme de Clasificación (Regresión Logística):")
print(class_report_log_reg)
```

#### **Interpretación de la Regresión Logística:**

- La **Accuracy** nos da el porcentaje total de predicciones correctas.
- La **Matriz de Confusión** nos muestra:
  - Verdaderos Negativos (TN): Casos Benignos predichos correctamente como Benignos.
  - Falsos Positivos (FP): Casos Benignos predichos incorrectamente como Malignos (Error Tipo I).
  - Falsos Negativos (FN): Casos Malignos predichos incorrectamente como Benignos (Error Tipo II - ¡muy importante en diagnóstico!).
  - Verdaderos Positivos (TP): Casos Malignos predichos correctamente como Malignos.
- El **Informe de Clasificación** detalla Precision, Recall y F1-score para cada clase.
  - **Precision (para Maligno):** De todos los que predijo como Malignos, ¿cuántos realmente lo eran? TP / (TP + FP)
  - **Recall (Sensibilidad, para Maligno):** De todos los que realmente eran Malignos, ¿cuántos detectó? TP / (TP + FN)
  - **F1-score:** Media armónica de Precision y Recall.

#### **Paso 5: Selección, Entrenamiento y Evaluación del Modelo 2 - Random Forest Classifier 🌳🌳**

**¿Por qué Random Forest?**

- Es un método ensemble (basado en árboles de decisión) que generalmente ofrece alta precisión.
- Es robusto al sobreajuste (overfitting) en comparación con un solo árbol de decisión.
- Puede capturar relaciones no lineales en los datos.
- No requiere que las características estén escaladas (aunque ya las tenemos escaladas, no le perjudica).

```python
print("\n--- Modelo 2: Random Forest Classifier ---")
# Instanciar el modelo
# n_estimators: número de árboles en el bosque. random_state para reproducibilidad.
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
# class_weight='balanced' puede ayudar si hay desbalanceo, aunque aquí es leve.

# Entrenar el modelo
rf_clf.fit(X_train_scaled, y_train) # Podríamos usar X_train sin escalar, pero con escalado está bien también
print("Modelo Random Forest entrenado. ✅")

# Realizar predicciones en el conjunto de prueba
y_pred_rf = rf_clf.predict(X_test_scaled)
y_pred_proba_rf = rf_clf.predict_proba(X_test_scaled)[:, 1] # Probabilidades para la clase positiva (1)

# Evaluar el modelo
accuracy_rf = accuracy_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
class_report_rf = classification_report(y_test, y_pred_rf)

print(f"\nAccuracy (Random Forest): {accuracy_rf:.4f}")
print(f"AUC-ROC (Random Forest): {roc_auc_score(y_test, y_pred_proba_rf):.4f}")
print("\nMatriz de Confusión (Random Forest):")
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión - Random Forest')
plt.show()

print("\nInforme de Clasificación (Random Forest):")
print(class_report_rf)
```

#### **Paso 6: Visualizaciones Adicionales 🖼️**

1. **Gráfico 1: Heatmap de Correlación de Características**

   Esto nos ayuda a entender las relaciones lineales entre las características. Lo haremos sobre el X original (antes de escalar y dividir) para tener una visión global.

```
print("\n--- Gráfico 1: Heatmap de Correlación de Características ---")
plt.figure(figsize=(18, 15)) # Ajustar tamaño para mejor visualización
# Usaremos X (features originales) para este heatmap
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".1f", linewidths=.5)
# annot=True puede ser muy denso con 30 features, por eso False. fmt=".1f" es para formato de anotación.
plt.title('Heatmap de Correlación de Características del Dataset Breast Cancer')
plt.show()
```

#### **Interpretación del Heatmap:**

- Colores rojos intensos indican correlación positiva fuerte.
- Colores azules intensos indican correlación negativa fuerte.
- Colores cercanos al blanco indican poca o ninguna correlación lineal.
- Podemos observar grupos de características altamente correlacionadas (multicolinealidad), lo cual es común en este dataset ya que muchas características son "mean", "stderr" y "worst" (peor o mayor) de las mismas mediciones base (ej. radius_mean, radius_stderr, radius_worst).

1. **Gráfico 2: Curvas ROC Comparativas**
   La curva ROC (Receiver Operating Characteristic) es excelente para comparar el rendimiento de clasificadores binarios, especialmente cuando las clases están desbalanceadas o los costos de los errores son diferentes. Grafica la Tasa de Verdaderos Positivos (Recall/Sensibilidad) contra la Tasa de Falsos Positivos.

```python
print("\n--- Gráfico 2: Curvas ROC Comparativas ---")
# Calcular FPR, TPR para Regresión Logística
fpr_log_reg, tpr_log_reg, thresholds_log_reg = roc_curve(y_test, y_pred_proba_log_reg)
auc_log_reg = roc_auc_score(y_test, y_pred_proba_log_reg)

# Calcular FPR, TPR para Random Forest
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_pred_proba_rf)
auc_rf = roc_auc_score(y_test, y_pred_proba_rf)

plt.figure(figsize=(10, 7))
plt.plot(fpr_log_reg, tpr_log_reg, label=f'Regresión Logística (AUC = {auc_log_reg:.3f})', color='blue', linestyle='--')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.3f})', color='green')
plt.plot([0, 1], [0, 1], color='red', linestyle=':', label='Clasificador Aleatorio (AUC = 0.500)') # Línea de no discriminación
plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad/Recall)')
plt.title('Comparación de Curvas ROC')
plt.legend()
plt.grid(True)
plt.show()
```

**Interpretación de la Curva ROC:**

- Un clasificador ideal tendría una curva que llega hasta la esquina superior izquierda (TPR=1, FPR=0).
- El Área Bajo la Curva (AUC) resume el rendimiento del clasificador. Un AUC de 1 es perfecto, 0.5 es aleatorio.
- Cuanto más "arriba y a la izquierda" esté la curva, mejor es el clasificador.
- Esta gráfica nos permite ver visualmente qué modelo tiene un mejor compromiso entre sensibilidad y especificidad en diferentes umbrales de decisión.

### **Paso 7: Conclusión y Discusión Final 🧑‍⚖️**

```
print("\n--- Conclusiones de la Práctica --- 🏁")

print(f"Resultados Regresión Logística: Accuracy = {accuracy_log_reg:.4f}, AUC = {auc_log_reg:.4f}")
print(f"Resultados Random Forest:       Accuracy = {accuracy_rf:.4f}, AUC = {auc_rf:.4f}")

print("\nComparación:")
if accuracy_rf > accuracy_log_reg and auc_rf > auc_log_reg:
    print("🏆 Random Forest mostró un mejor rendimiento general (Accuracy y AUC).")
elif accuracy_log_reg > accuracy_rf and auc_log_reg > auc_rf:
    print("🏆 Regresión Logística mostró un mejor rendimiento general (Accuracy y AUC).")
else:
    print("📊 Los modelos tuvieron rendimientos mixtos o muy similares en Accuracy y AUC. Revisar métricas específicas.")

print("\nDiscusión Detallada:")
print("Ambos modelos, Regresión Logística y Random Forest, lograron un buen rendimiento en la clasificación de tumores mamarios.")
print("1.  **Regresión Logística:** Si bien es un modelo más simple, obtuvo una alta accuracy y AUC, demostrando ser efectivo para este problema. Su interpretabilidad (a través de los coeficientes, no explorados aquí) podría ser una ventaja en contextos médicos.")
print("2.  **Random Forest:** Generalmente, se espera que Random Forest supere o iguale a modelos más simples. En este caso, si el AUC o la accuracy son ligeramente superiores, podría ser la elección preferida si la máxima capacidad predictiva es el objetivo principal, incluso a costa de una menor interpretabilidad directa en comparación con la Regresión Logística.")
print("\nConsideraciones Cruciales para este Caso (Médico):")
print("   - **Falsos Negativos (FN):** En un diagnóstico de cáncer, un Falso Negativo (predecir 'Benigno' cuando es 'Maligno') es típicamente el error más costoso. Debemos prestar especial atención al Recall de la clase 'Maligno'.")
print("     - Recall para Maligno (Reg. Logística):", class_report_log_reg.splitlines()[3].split()[3]) # Extrae el recall de la clase 1
print("     - Recall para Maligno (Random Forest):", class_report_rf.splitlines()[3].split()[3])

print("\nPosibles Próximos Pasos:")
print("   - **Ajuste de Hiperparámetros:** Usar técnicas como GridSearchCV o RandomizedSearchCV para encontrar los mejores hiperparámetros para cada modelo.")
print("   - **Ingeniería de Características:** Aunque este dataset es bastante bueno, en otros casos se podrían crear nuevas características.")
print("   - **Explorar otros modelos:** Probar SVM, Gradient Boosting (XGBoost, LightGBM), Redes Neuronales.")
print("   - **Validación Cruzada más robusta:** Para una estimación más fiable del rendimiento.")
print("   - **Análisis de Importancia de Características (para Random Forest):** Ver qué características contribuyen más a la predicción.")

print("\n¡Práctica completada! 🎉 Has construido y evaluado dos modelos de ML para un problema de clasificación real.")
```