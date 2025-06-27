# 🧪 **Práctica en Google Colab: Comprendiendo la Matriz de Confusión con Datos Climáticos**

### 🔍 Tema: Clasificación de eventos extremos climáticos

**Objetivo:** Aprender a construir, interpretar y analizar una matriz de confusión usando un conjunto de datos climáticos reales.

### 🔗 **Paso 1: Cargar los datos climáticos**

Usaremos un dataset abierto de NOAA que contiene registros de eventos extremos en EE.UU.

```python
import pandas as pd

# 📂 URL del dataset de eventos extremos NOAA
url = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/StormEvents_details-ftp_v1.0_d2023_c20240311.csv.gz"

# ⬇️ Cargar una muestra de los datos
df = pd.read_csv(url, compression='gzip', low_memory=False)

# 👀 Ver las primeras filas
df.head()
```

### 📊 **Paso 2: Preprocesar los datos**

Nos enfocaremos en clasificar si un evento es "EXTREME" o "NO EXTREME" basándonos en la magnitud.

```python
# ✅ Nos quedamos con columnas útiles
df_filtered = df[['EVENT_TYPE', 'DAMAGE_PROPERTY', 'INJURIES_DIRECT']].dropna()

# 🧹 Limpieza: convertir daño a número
def convert_damage(damage):
    if isinstance(damage, str):
        if damage.endswith('K'):
            return float(damage[:-1]) * 1_000
        elif damage.endswith('M'):
            return float(damage[:-1]) * 1_000_000
        elif damage.endswith('B'):
            return float(damage[:-1]) * 1_000_000_000
    return 0

df_filtered['DAMAGE_PROPERTY'] = df_filtered['DAMAGE_PROPERTY'].apply(convert_damage)

# 🎯 Etiquetamos: evento extremo si daño o heridos supera umbral
df_filtered['EXTREME'] = ((df_filtered['DAMAGE_PROPERTY'] > 1_000_000) | 
                          (df_filtered['INJURIES_DIRECT'] > 10)).astype(int)

df_filtered[['EVENT_TYPE', 'DAMAGE_PROPERTY', 'INJURIES_DIRECT', 'EXTREME']].head()
```

### 🤖 **Paso 3: Crear un clasificador simple**

Simularemos un modelo que predice "EXTREME" solo si el tipo de evento es conocido por ser grave.

```python
# 🔍 Reglas basadas en experiencia
extreme_events = ['Tornado', 'Hurricane', 'Flash Flood', 'Heat']

# 📦 Simulamos predicciones
df_filtered['PREDICTED'] = df_filtered['EVENT_TYPE'].apply(lambda x: 1 if x in extreme_events else 0)

df_filtered[['EVENT_TYPE', 'EXTREME', 'PREDICTED']].sample(5)
```

### 🧩 **Paso 4: Generar la matriz de confusión**

Ahora veremos cómo se desempeñó nuestro “modelo manual”.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 🎯 Calcular la matriz
y_true = df_filtered['EXTREME']
y_pred = df_filtered['PREDICTED']
cm = confusion_matrix(y_true, y_pred)

# 📈 Mostrarla visualmente
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["NO EXTREME", "EXTREME"])
disp.plot(cmap="Blues")
```

### 📚 **Paso 5: Interpretar la matriz de confusión**

```python
tn, fp, fn, tp = cm.ravel()

print(f"✅ Verdaderos Negativos (TN): {tn}")
print(f"❌ Falsos Positivos (FP): {fp}")
print(f"⚠️ Falsos Negativos (FN): {fn}")
print(f"🎯 Verdaderos Positivos (TP): {tp}")
```

**Explicación:**

- ✅ **TN**: El modelo predijo "NO EXTREME" y acertó.
- ❌ **FP**: Predijo "EXTREME", pero era falso.
- ⚠️ **FN**: No detectó un evento extremo real.
- 🎯 **TP**: Predijo correctamente un evento extremo.

### 🧠 **Reflexión final**

> ¿Qué tan confiable fue nuestro clasificador simple? ¿Qué errores fueron más comunes?
>  Si este modelo se usara para alertas públicas, ¿qué implicaciones tendrían los falsos negativos?