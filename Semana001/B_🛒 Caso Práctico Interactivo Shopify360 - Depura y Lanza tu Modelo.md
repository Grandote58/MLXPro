# **🛒 Caso Práctico Interactivo: "Shopify360 - Depura y Lanza tu Modelo"**

## 🎯 Objetivos del Caso

1. Aplicar un flujo completo de preprocesamiento sobre un dataset realista del eCommerce.
2. Comprender el impacto de los valores perdidos, duplicados, atípicos y mal etiquetados en un modelo de ML.

## 📖 Alcance del Aprendizaje

Al finalizar esta actividad:

- El estudiante dominará técnicas clave de limpieza y transformación de datos en Python.
- Será capaz de documentar su flujo de limpieza con comentarios y buenas prácticas.
- Podrá dejar listo un dataset para usar en modelos de clasificación con Scikit-learn o TensorFlow.

## 📒 Nombre del Notebook Sugerido

```python
shopify360_preprocesamiento.ipynb
```

## 💡 Paso a Paso del Caso Práctico: Checklist en Acción

### 1. 💾 Cargar el dataset

```python
import pandas as pd
import numpy as np

df = pd.read_csv("clientes_ejemplo_eda.csv")
df.head()
```

### 2. 🔍 Revisar estructura y tipos de datos

```python
df.info()
df.describe(include='all')
```

### 3. 🔢 Detectar valores faltantes

```python
df.isnull().sum()
```

### 4. 🛠️ Imputar valores nulos (edad y genero)

```python
df['edad'].fillna(df['edad'].mean(), inplace=True)
df['genero'].fillna('No especificado', inplace=True)
```

### 5. 🔄 Detectar y eliminar duplicados

```python
df.duplicated().sum()
df.drop_duplicates(inplace=True)
```

### 6. 📝 Homogeneizar campos de texto

```python
df['genero'] = df['genero'].str.lower().str.strip()
df['compro'] = df['compro'].str.lower().str.strip()
df['compro'] = df['compro'].replace({'¿si?': 'si', 'sí': 'si'})
```

### 7. 🚨 Detectar outliers por ingreso mensual

```python
q1 = df['ingreso_mensual'].quantile(0.25)
q3 = df['ingreso_mensual'].quantile(0.75)
iqr = q3 - q1
limite_inf = q1 - 1.5 * iqr
limite_sup = q3 + 1.5 * iqr

df = df[(df['ingreso_mensual'] >= limite_inf) & (df['ingreso_mensual'] <= limite_sup)]
```

### 8. 🔄 Validar estructura final

```python
df.info()
df.isnull().sum()
df.head()
```

### 9. 📂 Exportar dataset limpio

```python
df.to_csv("clientes_limpio.csv", index=False)
from google.colab import files
files.download("clientes_limpio.csv")
```

------

## 🎓 Actividad Evaluativa (opcional)

- Explica en una celda Markdown por qué decidiste imputar edad con media y no con mediana.
- Crea una gráfica (histograma o boxplot) que muestre la distribución del ingreso antes y después del tratamiento de outliers.

------

## 📊 Conclusión del Caso

Este ejercicio ha demostrado el poder del preprocesamiento como paso clave en proyectos de Machine Learning. Shopify360 no solo tiene un dataset limpio, sino listo para entrenar modelos que predigan la recompra de clientes 🚀.

¡Estás listo para pasar al siguiente nivel: modelado predictivo!