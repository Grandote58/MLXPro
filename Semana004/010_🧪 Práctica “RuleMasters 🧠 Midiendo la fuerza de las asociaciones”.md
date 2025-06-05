# 🧪 **Práctica: “RuleMasters 🧠: Midiendo la fuerza de las asociaciones”**

### 🎯 **Objetivo**

Aplicar reglas de asociación sobre un conjunto real de transacciones de libros electrónicos y **comparar soporte, confianza y lift** para determinar cuál métrica resulta más útil en la toma de decisiones.

## 📂 **Dataset real utilizado**

- 📚 **Nombre:** *Online Retail Dataset*
- 🌐 **URL (Kaggle):**
   👉 https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
- 📁 Archivo: `online_retail_II.csv`
- 🛍️ Contiene miles de transacciones en línea de una tienda europea entre 2009 y 2011.

## 👨‍🏫 **Paso a paso en Google Colab**



### 🔹 1. Instalar librerías necesarias

```python
# ⚙️ Instalar librería de minería de reglas
!pip install mlxtend --quiet
```

### 🔹 2. Importar librerías

```python
# 📚 Librerías para asociación y análisis
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
```

### 🔹 3. Cargar y limpiar el dataset

🔽 **Sube `online_retail_II.csv` desde tu equipo a Colab**

```python
# 📂 Cargar archivo
df = pd.read_csv("online_retail_II.csv", encoding='ISO-8859-1')
df = df.dropna(subset=['InvoiceNo', 'Description'])  # 🚿 Quitar registros con datos nulos

# 🧼 Filtrar transacciones reales (solo ventas, no devoluciones)
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
df.head()
```

### 🔹 4. Transformar a formato transaccional

```python
# 🧺 Agrupar descripciones por factura
basket = df.groupby(['InvoiceNo'])['Description'].apply(list)

# 🔄 Codificar datos binarios (presencia/ausencia del producto)
te = TransactionEncoder()
basket_encoded = te.fit_transform(basket)
df_encoded = pd.DataFrame(basket_encoded, columns=te.columns_)
df_encoded.head()
```

### 🔹 5. Aplicar Apriori para obtener itemsets frecuentes

```python
# 📊 Usamos un soporte mínimo del 1%
frequent_items = apriori(df_encoded, min_support=0.01, use_colnames=True)
frequent_items.sort_values(by='support', ascending=False).head()
```

### 🔹 6. Generar reglas de asociación

```python
# 🔗 Generar todas las reglas
rules = association_rules(frequent_items, metric="support", min_threshold=0.01)

# 🎯 Filtrar por confianza y lift altos
rules_filtered = rules[(rules['confidence'] >= 0.3) & (rules['lift'] >= 1.0)]
rules_filtered.head()
```

### 🔹 7. Comparar métricas: Soporte vs Confianza vs Lift

```python
# 📈 Visualización cruzada de métricas
plt.figure(figsize=(10,6))
plt.scatter(rules_filtered['support'], rules_filtered['confidence'], alpha=0.6, label='Soporte vs Confianza')
plt.scatter(rules_filtered['support'], rules_filtered['lift'], alpha=0.6, label='Soporte vs Lift')
plt.title("Comparación de métricas de reglas 🧠")
plt.xlabel("Soporte")
plt.ylabel("Confianza / Lift")
plt.legend()
plt.grid(True)
plt.show()
```

## 🔍 **Análisis y Conclusión**

| Métrica       | Ventajas                        | Desventajas                           | ¿Cuándo usarla?                      |
| ------------- | ------------------------------- | ------------------------------------- | ------------------------------------ |
| **Soporte**   | Objetiva y directa              | Penaliza productos poco frecuentes    | Cuando interesa lo común o frecuente |
| **Confianza** | Interpretable como probabilidad | Puede ser alta por coincidencia       | Si importa la certeza condicional    |
| **Lift**      | Relación ajustada al azar       | Puede sobrevalorar asociaciones raras | Para medir fuerza real de asociación |

## 🧠 **Reflexión**

1. ¿Qué reglas tienen alto *lift* pero bajo *soporte*? ¿Son útiles?
2. ¿Cuál métrica te da más confianza para recomendar productos?
3. ¿Qué métrica usarías en una tienda digital con artículos muy variados?

## 📎 **Conclusión**

✅ Aplicaste minería de reglas sobre datos reales

✅ Visualizaste y comparaste *soporte*, *confianza* y *lift*

✅ Evaluaste qué métrica aporta más valor según el contexto

📊 *“Más allá de descubrir patrones, debemos entender su relevancia.”*