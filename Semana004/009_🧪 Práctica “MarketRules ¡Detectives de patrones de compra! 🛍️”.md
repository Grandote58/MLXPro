# 🧪 **Práctica: “MarketRules: ¡Detectives de patrones de compra! 🛍️”**

### 🎯 **Objetivo**

Aplicar el algoritmo **Apriori** para descubrir **reglas de asociación** en un dataset real de transacciones de supermercado. Aprenderás a identificar patrones como:

> “Los clientes que compran pan 🍞 también tienden a comprar mantequilla 🧈”.

## 📂 **Dataset real utilizado**

- 📄 **Nombre:** *Groceries dataset*
- 🌐 **URL de descarga directa (Kaggle):**
   👉 https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset
- 📁 Archivo: `Groceries_dataset.csv`
- 🛒 Contiene más de 38,000 productos adquiridos en transacciones reales.

## 👨‍🏫 **Paso a paso en Google Colab (documentado con emojis)**

### 🔹 1. Instalar librerías necesarias

```python
# ⚙️ Instalar mlxtend si aún no lo tienes
!pip install mlxtend --quiet
```

### 🔹 2. Importar librerías

```python
# 📚 Librerías de análisis y minería de reglas
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
```

### 🔹 3. Cargar el dataset

🔽 **Sube `Groceries_dataset.csv` desde tu equipo a Google Colab**

```python
# 📂 Cargar archivo
df = pd.read_csv("Groceries_dataset.csv")
df.head()
```

### 🔹 4. Transformar el dataset a formato transaccional

```python
# 👀 Revisar estructura
print("Número de transacciones únicas:", df['Member_number'].nunique())

# 🧺 Agrupar productos por transacción
basket = df.groupby(['Date'])['itemDescription'].apply(list)

# 🧬 Codificar transacciones a formato binario
te = TransactionEncoder()
basket_encoded = te.fit(basket).transform(basket)
df_encoded = pd.DataFrame(basket_encoded, columns=te.columns_)
df_encoded.head()
```

### 🔹 5. Aplicar algoritmo Apriori

```python
# 📊 Extraer itemsets frecuentes con soporte mínimo de 1%
frequent_items = apriori(df_encoded, min_support=0.01, use_colnames=True)
frequent_items.sort_values(by='support', ascending=False).head()
```

### 🔹 6. Generar reglas de asociación

```python
# 🔗 Generar reglas a partir de los itemsets frecuentes
rules = association_rules(frequent_items, metric="lift", min_threshold=1.0)

# 📋 Filtrar reglas con buena confianza y lift
rules_filtered = rules[(rules['confidence'] >= 0.3) & (rules['lift'] >= 1.2)]

# 🔎 Ver las reglas más fuertes
rules_filtered.sort_values(by='lift', ascending=False).head()
```

### 🔹 7. Visualizar reglas más comunes

```python
# 📈 Visualizar por confianza vs lift
plt.figure(figsize=(8,6))
plt.scatter(rules_filtered['confidence'], rules_filtered['lift'], alpha=0.7)
plt.title("Visualización de Reglas 🧠")
plt.xlabel("Confianza")
plt.ylabel("Lift")
plt.grid(True)
plt.show()
```

### 🧠 **Reflexión**

1. ¿Qué productos aparecen con mayor frecuencia?
2. ¿Cuál es la regla más interesante por su *lift*?
3. ¿Cómo podrías usar esta información en una tienda real?

## 📎 **Conclusión**

✅ Aprendiste a transformar datos transaccionales

✅ Usaste el algoritmo **Apriori** para descubrir patrones

✅ Interpretaste las métricas *soporte, confianza y lift*

✅ Visualizaste las reglas más relevantes

🛍️ *“Descubrir patrones de compra es el primer paso para predecir comportamientos.”*