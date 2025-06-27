





# **Caso de Estudio** 

## 🚑 Predicción de insuficiencia cardíaca

Nuestro objetivo es construir un modelo que, a partir de datos clínicos de un paciente, prediga si sufrirá un evento mortal (insuficiencia cardíaca).

## **El Objetivo de Aprendizaje 🧠:**

No solo calcular métricas, sino **interpretar visualmente** qué significa cada una en un contexto donde los errores tienen consecuencias reales.

### **✅ Paso 0: Preparando nuestro Laboratorio Virtual**

Primero, importamos las herramientas que necesitaremos. `pandas` para manejar los datos, `scikit-learn` para el modelo y las métricas, y `matplotlib`/`seaborn` para las visualizaciones.

```python
# Importación de librerías esenciales
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_curve, 
    roc_auc_score,
    classification_report
)

# Estilo visual para nuestros gráficos
plt.style.use('seaborn-v0_8-whitegrid')
```

### **✅ Paso 1: Cargando y Explorando los Datos del Paciente 🔬**

Cargaremos el dataset directamente desde una URL. Esto es genial porque no necesitas descargar nada. Es un dataset público y limpio, ideal para aprender.

**Target a Predecir:** DEATH_EVENT **(0 = No murió, 1 = Murió)**

```python
# URL del dataset en formato raw
url = 'https://raw.githubusercontent.com/dataspelunking/MLwHeartFailure/main/heart_failure_clinical_records_dataset.csv'

# Cargar los datos en un DataFrame de pandas
df_pacientes = pd.read_csv(url)

# Echemos un primer vistazo a los datos de nuestros pacientes
print("Primeros 5 registros de pacientes:")
display(df_pacientes.head())

# 📊 ¡Importante! Verifiquemos el balance de clases
print("\nDistribución de la variable objetivo (DEATH_EVENT):")
print(df_pacientes['DEATH_EVENT'].value_counts())
sns.countplot(x='DEATH_EVENT', data=df_pacientes)
plt.title('Distribución de Clases: ¿Dataset Balanceado?')
plt.show()
```

> **Interpretación:** Vemos que el dataset está **desbalanceado**. Hay muchos más pacientes que sobrevivieron (Clase 0) que los que no (Clase 1). 
>
> ¡Esto hace que la métrica de 'Exactitud' por sí sola sea **peligrosa** y nos obliga a usar otras métricas!

### ✅ Paso 2: Pre-procesamiento y Entrenamiento del Modelo ❤️

Vamos a preparar los datos y entrenar un modelo simple pero efectivo: la Regresión Logística.

1. **Separar** características (X) de la etiqueta que queremos predecir (y).
2. **Dividir** los datos en un set de entrenamiento (para que el modelo aprenda) y un set de prueba (para evaluarlo con datos que nunca ha visto).
3. **Entrenar** el modelo.



```python
# 1. Separar características (X) y objetivo (y)
X = df_pacientes.drop('DEATH_EVENT', axis=1)
y = df_pacientes['DEATH_EVENT']

# 2. Dividir en datos de entrenamiento y prueba (80% / 20%)
# Usamos 'stratify=y' para mantener la misma proporción de clases en ambos sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# 3. Escalar los datos es una buena práctica para muchos modelos
# (No lo haremos aquí para mantener la simplicidad, pero es importante saberlo)

# 4. Crear y entrenar el modelo de Regresión Logística
modelo = LogisticRegression(max_iter=1000, random_state=42) # Aumentamos max_iter para asegurar convergencia
modelo.fit(X_train, y_train)
```

### ✅ Paso 3: Obteniendo las Predicciones 🎯

Ahora que el modelo está entrenado, lo usaremos para predecir los resultados en nuestro conjunto de prueba.

```python
# El modelo ahora predice sobre los datos de prueba
y_pred = modelo.predict(X_test)

# También obtenemos las probabilidades de predicción para la curva ROC
y_pred_proba = modelo.predict_proba(X_test)[:, 1] # Probabilidad de la clase '1'

# Guardemos los valores reales y predichos para facilitar el acceso
valores_reales = y_test
predicciones = y_pred
```

### ✅ Paso 4: ¡La Hora de la Verdad! Métrica por Métrica 📊

Aquí es donde interrogamos a nuestro modelo.

#### **4.1 Matriz de Confusión: El Mapa de la Verdad**

Este es el punto de partida de todo. 

Nos muestra los 4 posibles resultados de una predicción binaria.

```python
# Calcular la matriz de confusión
cm = confusion_matrix(valores_reales, predicciones)

# Extraer los valores para una mejor explicación
TN, FP, FN, TP = cm.ravel()

print(f"Verdaderos Negativos (TN): {TN} -> El modelo predijo 'No Muerte' y acertó.")
print(f"Falsos Positivos (FP): {FP} -> El modelo predijo 'Muerte' pero el paciente sobrevivió. (Falsa Alarma 🚨)")
print(f"Falsos Negativos (FN): {FN} -> El modelo predijo 'No Muerte' pero el paciente murió. (¡El peor error! 🚑)")
print(f"Verdaderos Positivos (TP): {TP} -> El modelo predijo 'Muerte' y acertó.")

# Visualización gráfica de la Matriz de Confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicho: No Muerte', 'Predicho: Muerte'], 
            yticklabels=['Real: No Muerte', 'Real: Muerte'])
plt.title('Matriz de Confusión: El Mapa de Aciertos y Errores', fontsize=15)
plt.ylabel('Valor Real')
plt.xlabel('Valor Predicho')
plt.show()
```

> **Interpretación Visual:** El gráfico nos muestra claramente los aciertos en la diagonal principal (azul oscuro). Vemos que hay **5 Falsos Negativos**, que son los casos más críticos que debemos intentar reducir.

#### **4.2 Exactitud (Accuracy): La Métrica General**

> ¿Qué porcentaje total de predicciones fue correcto?

```python
# Calcular la exactitud
accuracy = accuracy_score(valores_reales, predicciones)
print(f"🎯 Exactitud (Accuracy): {accuracy:.2%}")

# Gráfico interpretativo
fig, ax = plt.subplots(figsize=(7, 3))
ax.barh(['Predicciones Correctas'], [accuracy], color='cornflowerblue', label=f'{accuracy:.2%}')
ax.barh(['Predicciones Incorrectas'], [1 - accuracy], left=[accuracy], color='salmon', label=f'{1 - accuracy:.2%}')
ax.set_xlim(0, 1)
ax.set_xticks([])
ax.set_title("Desglose de la Exactitud")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
```

> **Interpretación:** Un 81.67% de exactitud parece bueno, pero ¡cuidado! Dado el desbalance de clases, esta métrica puede ser engañosa. Necesitamos investigar más a fondo.

#### **4.3 Precisión (Precision): La Calidad de las Alertas**

> De todas las veces que predijimos "Muerte", ¿qué porcentaje fue correcto?

```python
# Calcular la precisión
precision = precision_score(valores_reales, predicciones)
print(f"🔍 Precisión (Precision): {precision:.2%}")
print("Esta métrica nos dice qué tan confiables son nuestras alarmas de 'Muerte'. Un 70% significa que de 10 alarmas, 3 son falsas.")
```

#### **4.4 Exhaustividad (Recall / Sensibilidad): El Poder de Detección**

> De todos los pacientes que *realmente* murieron, ¿a cuántos logramos detectar?

```python
# Calcular la exhaustividad (recall)
recall = recall_score(valores_reales, predicciones)
print(f"❤️ Exhaustividad (Recall / Sensibilidad): {recall:.2%}")
print("¡Esta es una métrica CRÍTICA aquí! Nos dice que solo encontramos al 74% de los pacientes en riesgo. ¡El 26% restante son Falsos Negativos!")
```

#### **4.5 Puntuación F1 (F1-Score): El Equilibrio**

> Una media armónica que combina Precisión y Recall. Útil cuando las clases están desbalanceadas.

```python
# Calcular el F1-Score
f1 = f1_score(valores_reales, predicciones)
print(f"⚖️ Puntuación F1 (F1-Score): {f1:.2%}")
print("Nos da un solo número que balancea la precisión y el recall. Es una mejor medida general que la exactitud para este problema.")
```

> **Gráfico Interpretativo (Precisión, Recall, F1):**

```python
# Gráfico comparativo de las métricas clave
metrics_dict = {'Precisión': precision, 'Recall': recall, 'F1-Score': f1}
metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index', columns=['Puntuación'])

ax = metrics_df.plot(kind='bar', figsize=(8, 5), legend=False, colormap='viridis')
plt.title('Comparativa: Precisión vs. Recall vs. F1-Score')
plt.ylabel('Puntuación')
plt.xticks(rotation=0)
ax.bar_label(ax.containers[0], fmt='%.2f')
plt.show()
```

> **Interpretación:** Este gráfico nos permite ver el famoso **trade-off**. Tenemos un Recall decente pero una Precisión un poco más baja. El F1-Score nos da un valor intermedio que resume este balance.

#### **4.6 Especificidad (Specificity): Identificando a los Sanos**

> De todos los pacientes que *realmente* estaban bien (no murieron), ¿a cuántos clasificamos correctamente?

```python
# La especificidad se calcula a partir de la matriz de confusión: TN / (TN + FP)
specificity = TN / (TN + FP)
print(f"🛡️ Especificidad (Specificity): {specificity:.2%}")
print("Nuestro modelo es bastante bueno (85%) identificando a los pacientes que no van a tener un evento mortal.")
```

#### **4.7 Curva ROC y AUC: La Visión Panorámica**

Esta curva muestra cómo se comporta el modelo en **todos los posibles umbrales de decisión**. El área bajo la curva (AUC) nos da una medida global de rendimiento.

```python
# Calcular FPR, TPR para la curva ROC
fpr, tpr, thresholds = roc_curve(valores_reales, y_pred_proba)
# Calcular el AUC
auc = roc_auc_score(valores_reales, y_pred_proba)

# Graficar la Curva ROC
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Clasificador Aleatorio')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
plt.ylabel('Tasa de Verdaderos Positivos (Recall / Sensibilidad)')
plt.title('Curva ROC: Rendimiento General del Clasificador', fontsize=15)
plt.legend(loc="lower right")
plt.show()
```

> **Interpretación Visual:** Nuestro modelo (línea naranja) está significativamente por encima de la línea de azar (azul punteada). Un AUC de 0.88 es bastante robusto e indica que el modelo tiene un buen poder de discriminación entre las dos clases. Cuanto más se acerque la curva a la esquina superior izquierda, mejor.

### ✅ Paso 5: Resumen y Conclusiones Finales 🎓

Scikit-learn nos da una herramienta fantástica para ver un resumen de todo.

```python
# Reporte de clasificación completo
reporte = classification_report(valores_reales, predicciones, target_names=['No Muerte', 'Muerte'])
print("📋 Reporte de Clasificación Completo:")
print(reporte)
```

**Conclusión :**

Hemos pasado de una simple "Exactitud" a un entendimiento profundo del comportamiento de nuestro modelo.

Sabemos que es bueno identificando a los sanos (**alta Especificidad**), pero tiene un margen de mejora para encontrar a todos los enfermos (**Recall del 74%**).

Dependiendo del objetivo (¿minimizar falsas alarmas o no dejar escapar a ningún enfermo?), podríamos ajustar el umbral de decisión del modelo para optimizar la Precisión o el Recall.

