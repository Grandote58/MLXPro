# 🎓 **Lección e-Learning: “🧠 La Gráfica del Codo: Descubriendo cuántos grupos hay realmente”**

### 🎯 **Objetivo de aprendizaje**

Al finalizar esta lección, el estudiante será capaz de:

- Comprender qué representa la gráfica del codo
- Aplicarla correctamente para seleccionar el número óptimo de clústeres
- Interpretarla con ejemplos reales
- Evaluar su utilidad y limitaciones en proyectos de análisis de datos

## 📘 1. ¿Qué es la gráfica del codo?

> La **gráfica del codo (Elbow Method)** es una herramienta visual que se utiliza para **determinar el número óptimo de clústeres** en algoritmos de agrupamiento no supervisado, especialmente en **K-Means**.

📌 **¿Por qué es importante?**
 Porque en algoritmos como K-Means, debemos definir cuántos grupos (k) queremos, pero no siempre lo sabemos de antemano. La gráfica del codo nos ayuda a **elegir k de manera informada y no arbitraria**.

## 📊 2. Concepto clave: WCSS

**WCSS = Within-Cluster Sum of Squares**
 Es la suma de las distancias cuadradas de cada punto respecto al centroide de su clúster.

> A medida que aumentamos k, el WCSS **disminuye** porque hay más grupos que explican mejor los datos… pero **llega un punto donde la mejora deja de ser significativa**.

Ese “punto de cambio” es el **codo** de la gráfica.

## 📈 3. ¿Cómo se ve una gráfica del codo?

**Visualización típica:**

```javascript
Y eje: WCSS (Error)
X eje: Número de clústeres (k)

Gráfico: Decreciente, con un cambio de pendiente claro → "codo"
```

🧠 Interpretación:

- Antes del codo: gran mejora por cada nuevo clúster
- Después del codo: mejora marginal → no vale la pena añadir más

📷 **Ejemplo visual (simulado):**

```css
WCSS
│    ●
│     ●
│       ●
│         ●    ←  CODO
│           ●
│            ●
└─────────────────────
         k=1   k=6
```

## 🔬 4. Características de la gráfica del codo

| Característica       | Descripción                                        |
| -------------------- | -------------------------------------------------- |
| 📉 **Decreciente**    | El error disminuye al aumentar el número de grupos |
| 🎯 **Codo**           | Cambio notable en la pendiente                     |
| 🤔 **Subjetiva**      | A veces no es fácil ver el codo claramente         |
| 📏 **Basada en WCSS** | Mide la compactación de los clústeres              |

## 🧪 5. Ejemplo práctico: Segmentación de clientes

**Caso:** Una tienda quiere segmentar a sus clientes según ingresos y frecuencia de compra.

🔧 Usamos K-Means con k de 1 a 10 → Calculamos WCSS
 📈 Trazamos la gráfica del codo → El codo aparece en **k=3**

✍️ **Conclusión práctica:**

> 3 clústeres es un buen número para agrupar clientes con comportamientos similares

### 📌 Código en Google Colab:

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Suponiendo datos escalados en X
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Graficar
plt.plot(range(1, 11), wcss, marker='o')
plt.title("📈 Método del Codo para encontrar k óptimo")
plt.xlabel("Número de clústeres (k)")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()
```

## ⚠️ 6. Limitaciones del método

- 📉 El “codo” puede ser difícil de interpretar
- ❌ No siempre hay un codo claro
- 🔀 No evalúa la separación entre grupos (solo la compactación)

💡 **Alternativas o complementos:**

- Índice de Silueta
- GAP Statistic
- Davies-Bouldin Index

## 📎 Conclusión

✅ Entendiste qué representa la gráfica del codo

✅ Aplicaste su uso en la selección de clústeres

✅ Identificaste sus ventajas y limitaciones

✅ Conociste un caso real de segmentación de clientes

📊 *“El codo no solo dobla la curva… ¡te señala la decisión correcta!”*