# **✨ Proyecto | Diagnóstico Inteligente: Más Allá de la Precisión en la Predicción de Enfermedades Cardíacas**



### **🧠 ¡Bienvenido/a, Data Scientist! 🧠**

Hoy no solo vamos a escribir código. 

Vamos a resolver un problema real que tiene un impacto directo en la vida de las personas.

## 🚀 El Reto: La Startup "CardioSafe Analytics"

Imagina que eres parte del equipo de Machine Learning de **"CardioSafe Analytics"**, una startup de HealthTech que ha desarrollado un modelo inicial para predecir la probabilidad de que un paciente sufra una enfermedad cardíaca.

El primer modelo, basado en una regresión logística simple, alcanzó una **precisión (accuracy) del 85%**. El equipo directivo está contento, pero el equipo médico está preocupado. Nos hacen una pregunta crucial:

> ***"Un 85% de precisión suena bien, pero... ¿qué significa realmente? ***
>
> ***¿A cuántos pacientes realmente enfermos estamos mandando a casa diciéndoles que están sanos? ***
>
> ***¿Y a cuántos sanos estamos asustando innecesariamente? ***
>
> ***Necesitamos entender la eficiencia real del modelo, no solo un número."***

Tu misión es tomar el control del proyecto. Deberás explorar los datos, proponer y entrenar un modelo más robusto, y lo más importante: **evaluar su eficiencia desde una perspectiva clínica**, explicando a los stakeholders (los médicos y directivos) qué significan realmente los resultados.

## 🎯 Objetivos de la Práctica



Al finalizar este reto, serás capaz de:

1. **Realizar un Análisis Exploratorio de Datos (EDA)** en un dataset de salud.
2. **Preprocesar los datos** para prepararlos para el entrenamiento de un modelo.
3. **Seleccionar, justificar e implementar un modelo de clasificación** robusto como Random Forest.
4. **Evaluar la eficiencia del modelo** utilizando métricas avanzadas más allá de la precisión, como la matriz de confusión, la precisión (precision) y la sensibilidad (recall).
5. **Interpretar los resultados del modelo** en el contexto de un problema real y comunicar tus hallazgos de forma efectiva.
6. **Identificar los factores más influyentes** en la predicción de la enfermedad.

## 🌐 El Dataset: Heart Disease UCI



Usaremos un dataset clásico y muy conocido en la comunidad de Machine Learning, alojado en un repositorio de GitHub para fácil acceso. Contiene 14 atributos extraídos de pacientes, como edad, sexo, nivel de colesterol, y si tienen o no una enfermedad cardíaca (`target`).

**URL del dataset (raw):** `https://raw.githubusercontent.com/dataspelunking/MLwHeartDisease/main/Data/processed.cleveland.data.csv`

## 💻 Desarrollo del Proyecto en Google Colab

¡Es hora de poner manos a la obra! Sigue estos pasos en tu notebook.

### 📘 Nota Didáctica: Fase 1 - Entendiendo a nuestros "Pacientes" Digitales

Antes de cualquier diagnóstico, un buen médico (y un buen Data Scientist) debe conocer a su paciente. Esta primera fase se llama **Análisis Exploratorio de Datos (EDA)**. Nuestro objetivo es "escuchar" lo que los datos nos dicen sobre las características de los pacientes y la distribución de la enfermedad.

#### **Paso 1.1: Configuración del Entorno e Importación de Librerías**

```python
# 🧠 Primero, importamos las herramientas que necesitaremos en nuestro laboratorio digital.
# pandas para manejar los datos como si fueran hojas de cálculo avanzadas.
# matplotlib y seaborn para crear visualizaciones que nos ayuden a "ver" los datos.
# scikit-learn para los modelos de Machine Learning y las métricas de evaluación.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Configuramos un estilo visual agradable para nuestros gráficos.
sns.set_style('whitegrid')
%matplotlib inline
```

#### **Paso 1.2: Carga y Primera Exploración de los Datos**

```python
# 🧠 Cargamos el dataset directamente desde la URL.
# Le asignamos nombres a las columnas según la documentación del dataset para que sea más legible.

url = 'https://raw.githubusercontent.com/dataspelunking/MLwHeartDisease/main/Data/processed.cleveland.data.csv'
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
# El dataset usa '?' para los valores faltantes, se lo indicamos a pandas.
df = pd.read_csv(url, header=None, names=column_names, na_values='?')

# 👀 Echamos un primer vistazo a los datos para asegurarnos de que se cargaron correctamente.
df.head()
```

-----

```python
# 🧠 Obtenemos información general del dataset.
# ¿Hay valores nulos? ¿Qué tipo de dato tiene cada columna?
# Esto es como la primera ficha médica del paciente.
df.info()
```

### 📘 Nota Didáctica: Fase 2 - Limpieza y Preparación del Quirófano

Los datos del mundo real rara vez son perfectos. A menudo tienen "heridas" (valores faltantes) que debemos curar antes de operar (entrenar el modelo). Esta fase de **limpieza y preprocesamiento** es crucial para la salud de nuestro modelo.

#### **Paso 2.1: Manejo de Datos Faltantes**

```python
# 🧠 Identificamos cuántos valores nulos hay por columna.
df.isnull().sum()
```

-------

```python
# 🧠 Dado que son pocos los valores nulos, una estrategia simple y efectiva es rellenarlos
# con la mediana de su respectiva columna. La mediana es más robusta a valores atípicos que la media.
# Es una "cirugía" menor y segura.

df['ca'].fillna(df['ca'].median(), inplace=True)
df['thal'].fillna(df['thal'].median(), inplace=True)

# Verificamos que ya no hay valores nulos. ¡El paciente está limpio!
print("Valores nulos después de la limpieza:")
print(df.isnull().sum())
```

#### **Paso 2.2: Visualizando la Distribución de la Enfermedad**

```python
# 📊 ¿Cuántos pacientes en nuestro dataset están enfermos vs. sanos?
# La columna 'target' nos lo dice: 0 = No hay enfermedad, >0 = Hay enfermedad.
# Para simplificar, convertiremos todos los valores > 0 a 1.

df['target'] = (df['target'] > 0).astype(int)

# Ahora visualizamos la distribución.
plt.figure(figsize=(8, 6))
sns.countplot(x='target', data=df, palette='viridis')
plt.title('Distribución de Pacientes con y sin Enfermedad Cardíaca')
plt.xticks([0, 1], ['Sanos (0)', 'Enfermos (1)'])
plt.ylabel('Cantidad de Pacientes')
plt.show()

print(df['target'].value_counts(normalize=True))
```

> **Interpretación:** Vemos que el dataset está razonablemente balanceado. No hay una clase abrumadoramente mayoritaria, lo cual es bueno para el entrenamiento del modelo.

#### **Paso 2.3: Visualizando la Correlación entre Variables**

```python
# 📊 Una matriz de correlación nos ayuda a entender cómo se relacionan las variables entre sí.
# ¿Qué factores están más asociados con la enfermedad cardíaca?
# Un mapa de calor es la mejor forma de visualizar esto.

plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de Calor de Correlación de Variables')
plt.show()
```

> **Interpretación:** Observa la última fila (`target`). Las variables con colores más intensos (rojo o azul) son las que tienen una mayor correlación (positiva o negativa) con la presencia de una enfermedad cardíaca. Por ejemplo, `cp` (tipo de dolor de pecho) y `thalach` (frecuencia cardíaca máxima) tienen una correlación positiva fuerte, mientras que `exang` (angina inducida por ejercicio) y `oldpeak` tienen una correlación negativa.

### 📘 Nota Didáctica: Fase 3 - Construyendo y Evaluando Nuestro Nuevo Modelo

Ahora viene el núcleo de nuestra misión. Separaremos los datos para entrenar y probar nuestro modelo. Elegiremos un **Random Forest** en lugar de la regresión logística inicial.

> **¿Por qué un Random Forest?**
>
> 1. **Es más robusto:** Es un "comité de expertos" (múltiples árboles de decisión) que votan por el resultado final. Esto reduce el riesgo de que un solo "experto" se equivoque (overfitting).
> 2. **Maneja relaciones complejas:** No asume que la relación entre las variables es lineal, algo muy común en biología.
> 3. **Nos da "Feature Importance":** Nos puede decir qué variables considera más importantes para hacer su diagnóstico, ¡información valiosísima para los médicos!

#### **Paso 3.1: Separación de Datos en Entrenamiento y Prueba**

```python
# 🧠 Separamos nuestro dataset en dos:
# X: Las características del paciente (las variables predictoras).
# y: El diagnóstico que queremos predecir (la variable objetivo, 'target').

X = df.drop('target', axis=1)
y = df['target']

# Dividimos los datos. Usaremos el 80% para entrenar al modelo y el 20% para probar
# su eficiencia de forma imparcial. `random_state` asegura que la división sea siempre la misma.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Tamaño del set de entrenamiento: {X_train.shape[0]} pacientes")
print(f"Tamaño del set de prueba: {X_test.shape[0]} pacientes")
```

#### **Paso 3.2: Entrenamiento del Modelo Random Forest**

```python
# 🧠 Creamos una instancia del clasificador Random Forest.
# n_estimators=100 significa que nuestro "comité" tendrá 100 árboles de decisión.
# random_state=42 para reproducibilidad.

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# ¡El momento de la verdad! Entrenamos el modelo con los datos de entrenamiento.
# El modelo está "estudiando" los casos de los pacientes para aprender los patrones.
rf_model.fit(X_train, y_train)
```

#### **Paso 3.3: Evaluación de la Eficiencia del Modelo**

```python
# 🧠 Ahora que el modelo ha sido entrenado, lo probamos con datos que nunca ha visto antes.
y_pred = rf_model.predict(X_test)

# 1. La métrica básica: Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy del Modelo Random Forest: {accuracy:.2f}")
print("-" * 50)

# 2. El reporte completo para los médicos: Classification Report
# Aquí vemos la precisión (precision), sensibilidad (recall) y F1-score.
print("📊 Reporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['Sano (Clase 0)', 'Enfermo (Clase 1)']))
print("-" * 50)

# 3. La herramienta de diagnóstico clave: La Matriz de Confusión
print("🔍 Matriz de Confusión:")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicción Sano', 'Predicción Enfermo'],
            yticklabels=['Real Sano', 'Real Enfermo'])
plt.title('Matriz de Confusión')
plt.ylabel('Valor Real')
plt.xlabel('Valor Predicho')
plt.show()
```

### 📘 Nota Didáctica: Fase 4 - Comunicando los Resultados a "CardioSafe"

¡Excelente trabajo! Nuestro modelo parece tener una accuracy similar al anterior, pero ahora tenemos herramientas mucho más poderosas para responder a las preocupaciones del equipo médico.

> **Pensemos como médicos al leer la Matriz de Confusión:**
>
> - **Verdaderos Positivos (VP):** 24. Pacientes enfermos que diagnosticamos correctamente como enfermos. ¡Genial!
> - **Verdaderos Negativos (VN):** 29. Pacientes sanos que diagnosticamos correctamente como sanos. ¡Perfecto!
> - **Falsos Positivos (FP):** 2. Pacientes sanos que diagnosticamos erróneamente como enfermos. Esto genera ansiedad y costos de pruebas innecesarias, pero no es fatal.
> - **Falsos Negativos (FN):** 6. **¡ESTE ES EL NÚMERO CRÍTICO!** Son pacientes enfermos que mandamos a casa diciéndoles que estaban sanos. Es el peor error posible en este contexto.

El **Recall** (o sensibilidad) para la clase "Enfermo" nos dice que detectamos al 80% de los pacientes que realmente tenían la enfermedad (`24 / (24 + 6)`). Nuestro objetivo como equipo sería reducir esos 6 Falsos Negativos, incluso si eso significa aumentar un poco los Falsos Positivos.

#### **Paso 4.1: Identificando los Factores Clave (Feature Importance)**

```python
# 🧠 Una de las grandes ventajas del Random Forest.
# Podemos preguntarle al modelo: "¿Qué factores fueron los más importantes para tomar tus decisiones?"

importances = rf_model.feature_importances_
feature_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_df = feature_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_df, palette='viridis')
plt.title('Importancia de cada Característica en la Predicción')
plt.xlabel('Importancia')
plt.ylabel('Característica')
plt.show()
```

> **Conclusión para los Stakeholders:** *"Equipo, nuestro nuevo modelo no solo mantiene una buena precisión, sino que ahora podemos ver que los factores más determinantes para el diagnóstico son el **tipo de dolor de pecho (cp)**, la **frecuencia cardíaca máxima alcanzada (thalach)** y la **depresión del segmento ST (oldpeak)**. *
>
> *Más importante aún, hemos identificado que nuestro principal desafío son los **Falsos Negativos**. Propongo que la siguiente fase del proyecto se centre en técnicas para minimizar este error específico, ya que tiene el mayor impacto clínico."*

## ✅ ¿Qué has aprendido en esta misión?

Si has llegado hasta aquí, ¡felicidades! No solo has construido un modelo de Machine Learning, has actuado como un verdadero Data Scientist en un proyecto con impacto real. Concretamente, has logrado:

- **Habilidad Técnica:** Implementaste un flujo de trabajo completo de Machine Learning: carga, limpieza, visualización, entrenamiento y evaluación.
- **Pensamiento Crítico:** Comprendiste por qué la **accuracy no es suficiente** y aprendiste a usar e interpretar la **matriz de confusión** y el **reporte de clasificación** para evaluar la eficiencia real de un modelo.
- **Justificación de Modelos:** Entendiste las ventajas de un modelo como **Random Forest** sobre otros más simples, especialmente su capacidad para darnos la importancia de las características.
- **Comunicación Efectiva:** Aprendiste a "traducir" métricas técnicas (como Falsos Negativos) en **riesgos y conclusiones de negocio** (impacto clínico) para una audiencia no especializada.
- **Visión de Proyecto:** Te posicionaste no como alguien que solo ejecuta código, sino como un **solucionador de problemas** que analiza resultados y propone los siguientes pasos estratégicos.