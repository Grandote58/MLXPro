# **Práctica Guiada: Explorando un E-commerce con Aprendizaje No Supervisado 🛍️🔍**

**Objetivo General:** Utilizar diversas técnicas de aprendizaje no supervisado para extraer insights valiosos de un dataset de transacciones de un e-commerce, comprendiendo el comportamiento del cliente, la estructura de los datos y patrones de compra.

**Plataforma:** Google Colab

**Dataset Seleccionado: "Online Retail II UCI"**

- **Descripción:** Contiene transacciones ocurridas entre 01/12/2009 y 09/12/2011 para un minorista online registrado y no registrado en el Reino Unido. Muchas empresas mayoristas venden principalmente a otras empresas.
- **URL de Descarga:** [https://archive.ics.uci.edu/ml/machine-learning-databases/00502/](https://www.google.com/url?sa=E&q=https%3A%2F%2Farchive.ics.uci.edu%2Fml%2Fmachine-learning-databases%2F00502%2F) (Usaremos el archivo online_retail_II.xlsx)
- **¿Por qué este dataset?**
  - Tiene información de clientes (para K-Means, PCA).
  - Tiene información de productos y transacciones (para Analítica de Asociación).
  - Podemos derivar secuencias de compra por cliente (para HMM, conceptualmente).
  - Es lo suficientemente grande y real para ser interesante.

## **PASO 0: Configuración del Entorno y Carga de Datos 🛠️**

```python
# PASO 0: Configuración y Carga de Datos

# Instalar librerías necesarias (si no están presentes)
!pip install openpyxl mlxtend hmmlearn scikit-learn pandas numpy matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
from hmmlearn import hmm # Para Modelos Ocultos de Markov
import random # Para Algoritmos Genéticos (ejemplo simple)

# Silenciar warnings para una salida más limpia (opcional)
import warnings
warnings.filterwarnings('ignore')

print("Librerías importadas exitosamente! 👍")

# Cargar el dataset
# Nota: La primera vez puede tardar un poco en descargar y cargar.
# Este dataset tiene dos hojas, 'Year 2009-2010' y 'Year 2010-2011'.
# Vamos a trabajar con la hoja 'Year 2010-2011' por simplicidad, pero podrías combinarlas.
try:
    xls = pd.ExcelFile('https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx')
    df_retail = pd.read_excel(xls, 'Year 2010-2011')
    print("Dataset cargado exitosamente! 🛒")
    print(f"El dataset tiene {df_retail.shape[0]} filas y {df_retail.shape[1]} columnas.")
except Exception as e:
    print(f"Error al cargar el dataset: {e}")
    print("Asegúrate de que la URL es correcta y tienes conexión a internet.")
    df_retail = pd.DataFrame() # Crear un DF vacío para que el resto del notebook no falle

# Visualizar las primeras filas y la información del DataFrame
if not df_retail.empty:
    print("\nPrimeras 5 filas del dataset:")
    print(df_retail.head())
    print("\nInformación del dataset:")
    df_retail.info()
    print("\nEstadísticas descriptivas básicas:")
    print(df_retail.describe(include='all'))
else:
    print("\nNo se pudo cargar el dataset. El resto de la práctica no podrá ejecutarse correctamente.")
```

**🤔 Pregunta **

- ¿Qué tipo de datos contiene cada columna? 
- ¿Hay valores nulos? 
- ¿Qué podrías inferir inicialmente sobre los clientes y sus compras?

## **PASO 1: Preprocesamiento de Datos 🧹**

- **Concepto:** Antes de aplicar cualquier algoritmo, es crucial limpiar y transformar los datos.
- **Tareas:**
  - Manejar valores nulos (especialmente en Customer ID).
  - Convertir InvoiceDate a formato datetime.
  - Crear una columna TotalPrice = Quantity * Price.
  - Eliminar transacciones con Quantity negativa (devoluciones) o Price cero si no son relevantes para el análisis actual.
  - Filtrar registros sin Customer ID ya que son clave para la segmentación y otros análisis centrados en el cliente.

```python
# PASO 1: Preprocesamiento de Datos

if not df_retail.empty:
    print("Iniciando preprocesamiento... ⏳")
    # Copia del dataframe para no modificar el original directamente
    df_processed = df_retail.copy()

    # Manejar valores nulos en Customer ID (eliminarlos para este análisis)
    df_processed.dropna(subset=['Customer ID'], inplace=True)
    print(f"Filas después de eliminar Customer ID nulos: {df_processed.shape[0]}")

    # Convertir Customer ID a entero (ya que los decimales no tienen sentido aquí)
    df_processed['Customer ID'] = df_processed['Customer ID'].astype(int)

    # Convertir InvoiceDate a datetime
    df_processed['InvoiceDate'] = pd.to_datetime(df_processed['InvoiceDate'])

    # Crear TotalPrice
    df_processed['TotalPrice'] = df_processed['Quantity'] * df_processed['Price']

    # Eliminar transacciones con cantidad negativa (devoluciones) o precio cero/negativo
    # y aquellas con TotalPrice negativo o cero si no representan una venta real.
    df_processed = df_processed[df_processed['Quantity'] > 0]
    df_processed = df_processed[df_processed['Price'] > 0]
    df_processed = df_processed[df_processed['TotalPrice'] > 0] # Asegura que las ventas sean positivas
    print(f"Filas después de limpiar Quantity y Price negativos/cero: {df_processed.shape[0]}")

    # Eliminar duplicados si existen
    df_processed.drop_duplicates(inplace=True)
    print(f"Filas después de eliminar duplicados: {df_processed.shape[0]}")

    print("\nPreprocesamiento básico completado! ✨")
    print("\nPrimeras filas del dataset procesado:")
    print(df_processed.head())
    print("\nInformación actualizada:")
    df_processed.info()
else:
    print("Dataset no cargado, saltando preprocesamiento.")
```

**🤔 Pregunta**

- ¿Por qué es importante eliminar las filas con Customer ID nulo para un análisis de segmentación de clientes?
- ¿Qué otras decisiones de preprocesamiento podrías haber tomado y por qué?

## **PASO 2: Segmentación de Clientes con K-Means 🧑‍🤝‍🧑**

- **Concepto:** Agrupar clientes con comportamientos de compra similares.
- **Ingeniería de Características (RFM):** Crearemos características que resuman el comportamiento del cliente. Un enfoque común es RFM:
  - **Recency (R):** ¿Cuán recientemente compró el cliente? (Menor es mejor)
  - **Frequency (F):** ¿Con qué frecuencia compra? (Mayor es mejor)
  - **Monetary (M):** ¿Cuánto gasta? (Mayor es mejor)

```python
# PASO 2: Segmentación de Clientes con K-Means

if not df_processed.empty:
    print("\nIniciando segmentación con K-Means... 🎯")

    # Crear características RFM
    # Tomaremos una fecha de referencia (snapshot_date) para calcular Recency
    # Usaremos el día después de la última fecha de factura en el dataset.
    snapshot_date = df_processed['InvoiceDate'].max() + pd.Timedelta(days=1)
    print(f"Fecha de referencia (snapshot_date) para RFM: {snapshot_date}")

    # Calcular RFM
    rfm_data = df_processed.groupby(['Customer ID']).agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days, # Recency
        'Invoice': 'nunique',                                   # Frequency (número de facturas únicas)
        'TotalPrice': 'sum'                                     # Monetary
    }).reset_index()

    # Renombrar columnas
    rfm_data.rename(columns={'InvoiceDate': 'Recency',
                             'Invoice': 'Frequency',
                             'TotalPrice': 'MonetaryValue'}, inplace=True)

    print("\nDataset RFM creado:")
    print(rfm_data.head())

    # Preprocesamiento para K-Means: Escalar características
    # K-Means es sensible a la escala de las características
    rfm_scaled = rfm_data.copy()
    scaler = StandardScaler()
    rfm_scaled[['Recency', 'Frequency', 'MonetaryValue']] = scaler.fit_transform(
        rfm_scaled[['Recency', 'Frequency', 'MonetaryValue']]
    )

    print("\nDataset RFM escalado:")
    print(rfm_scaled.head())

    # Encontrar el número óptimo de clusters (K) usando el método del codo y la silueta
    sse = {} # Sum of Squared Errors (para el método del codo)
    silhouette_scores = {} # Para el coeficiente de silueta

    # Consideraremos de 2 a 10 clusters
    K_range = range(2, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(rfm_scaled[['Recency', 'Frequency', 'MonetaryValue']])
        sse[k] = kmeans.inertia_ # Inertia: Suma de las distancias al cuadrado de las muestras a su centro de clúster más cercano.
        silhouette_scores[k] = silhouette_score(rfm_scaled[['Recency', 'Frequency', 'MonetaryValue']], kmeans.labels_)

    # Graficar el método del codo
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(list(sse.keys()), list(sse.values()), marker='o')
    plt.xlabel("Número de Clusters (K)")
    plt.ylabel("SSE (Inertia)")
    plt.title("Método del Codo para K-Means")
    plt.grid(True)

    # Graficar el coeficiente de silueta
    plt.subplot(1, 2, 2)
    plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o')
    plt.xlabel("Número de Clusters (K)")
    plt.ylabel("Coeficiente de Silueta")
    plt.title("Coeficiente de Silueta para K-Means")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Elegir K (basado en los gráficos, supongamos K=4 como un buen compromiso)
    # El "codo" podría estar en 3 o 4. La silueta podría ser máxima en 2 o 3.
    # Vamos a elegir K=4 para tener más segmentos. ¡Anima al estudiante a probar otros K!
    optimal_k = 4
    print(f"\nBasado en el análisis (o elección pedagógica), usaremos K = {optimal_k}")

    # Aplicar K-Means con el K óptimo
    kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    rfm_data['Cluster'] = kmeans_final.fit_predict(rfm_scaled[['Recency', 'Frequency', 'MonetaryValue']])

    print("\nPrimeras filas del dataset RFM con clusters asignados:")
    print(rfm_data.head())

    # Analizar los centroides de los clusters (en la escala original, no la escalada)
    # Para ello, aplicamos inverse_transform a los centroides escalados o calculamos la media por cluster
    cluster_analysis = rfm_data.groupby('Cluster')[['Recency', 'Frequency', 'MonetaryValue']].mean().reset_index()
    print("\nAnálisis de los centroides de los clusters (valores promedio por cluster):")
    print(cluster_analysis)

    # Visualizar los clusters (usaremos las características originales para interpretabilidad)
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=rfm_data, x='Recency', y='MonetaryValue', hue='Cluster', palette='viridis', s=50, alpha=0.7)
    plt.title('Clusters de Clientes (Recency vs MonetaryValue)')
    plt.xlabel('Recencia (días desde la última compra)')
    plt.ylabel('Valor Monetario Total Gastado')
    # Marcar los centroides (calculados sobre los datos originales para la visualización)
    centroids_orig = rfm_data.groupby('Cluster')[['Recency', 'Frequency', 'MonetaryValue']].mean()
    plt.scatter(centroids_orig['Recency'], centroids_orig['MonetaryValue'], marker='X', s=200, color='red', label='Centroides')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=rfm_data, x='Frequency', y='MonetaryValue', hue='Cluster', palette='viridis', s=50, alpha=0.7)
    plt.title('Clusters de Clientes (Frequency vs MonetaryValue)')
    plt.xlabel('Frecuencia de Compra')
    plt.ylabel('Valor Monetario Total Gastado')
    plt.scatter(centroids_orig['Frequency'], centroids_orig['MonetaryValue'], marker='X', s=200, color='red', label='Centroides')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("K-Means completado y visualizado! 📊")

else:
    print("Dataset RFM no creado debido a problemas previos, saltando K-Means.")
```

**🤔 Pregunta**

- Interpreta los gráficos del método del codo y la silueta. ¿Qué valor de K habrías elegido y por qué?

- Analiza la tabla cluster_analysis. 

  ¿Qué características definen a cada cluster?

  (Ej: Cluster 0 = clientes recientes, de alto valor y frecuentes). Dales nombres descriptivos (ej. "Campeones", "Leales", "En Riesgo", "Nuevos").

- ¿Qué estrategias de marketing podrías proponer para cada segmento de clientes identificado?

## **PASO 3: Reducción de Dimensionalidad con PCA 📉**

- **Concepto:** Reducir el número de características (RFM) a 2 componentes principales para visualización, manteniendo la mayor varianza posible.
- **Aplicación:** Usaremos PCA sobre el dataset RFM escalado.

```python
# PASO 3: Reducción de Dimensionalidad con PCA

if 'rfm_scaled' in locals() and not rfm_scaled.empty:
    print("\nIniciando reducción de dimensionalidad con PCA... 🌀")

    # Aplicar PCA para reducir a 2 componentes
    pca = PCA(n_components=2, random_state=42)
    rfm_pca = pca.fit_transform(rfm_scaled[['Recency', 'Frequency', 'MonetaryValue']])

    # Crear un DataFrame con los componentes principales
    df_pca = pd.DataFrame(data=rfm_pca, columns=['PC1', 'PC2'])
    # Añadir la información del cluster para colorear la visualización
    df_pca['Cluster'] = rfm_data['Cluster'] # Asumiendo que rfm_data y rfm_scaled tienen el mismo orden de CustomerID

    print("\nDataset RFM reducido a 2 Componentes Principales:")
    print(df_pca.head())

    # Varianza explicada por cada componente
    print(f"\nVarianza explicada por PC1: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"Varianza explicada por PC2: {pca.explained_variance_ratio_[1]:.2%}")
    print(f"Varianza total explicada por 2 CPs: {pca.explained_variance_ratio_.sum():.2%}")

    # Visualizar los datos en el espacio de los Componentes Principales
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='viridis', s=50, alpha=0.7)
    plt.title('Clientes en el Espacio de Componentes Principales (PCA)')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend(title='Cluster K-Means')
    plt.grid(True)
    plt.show()

    print("PCA completado y visualizado! ✨")

    # (Opcional) Mostrar el gráfico de varianza explicada acumulada si se probaran más componentes
    pca_full = PCA(random_state=42)
    pca_full.fit(rfm_scaled[['Recency', 'Frequency', 'MonetaryValue']])
    explained_variance_ratio_cumulative = np.cumsum(pca_full.explained_variance_ratio_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_variance_ratio_cumulative) + 1), explained_variance_ratio_cumulative, marker='o', linestyle='--')
    plt.title('Varianza Explicada Acumulada por Componentes Principales')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Varianza Explicada Acumulada')
    plt.grid(True)
    plt.axhline(y=0.90, color='r', linestyle='-', label='90% Varianza') # Umbral común
    plt.legend()
    plt.show()

else:
    print("Dataset RFM escalado no disponible, saltando PCA.")
```

**🤔 Preguntas**

- ¿Cuánta varianza total explican los dos primeros componentes principales? 

- ¿Consideras que es suficiente para representar bien los datos?

- Observa la visualización de los clusters en el espacio PCA. 

  ​	¿Se separan bien los clusters? 

  ​	¿Cómo se compara esta visualización con las de Recency-Monetary y Frequency-Monetary?

- Si tuvieras muchas más características (ej. 10 o 20), ¿cuál sería la principal ventaja de usar PCA antes de K-Means?

## **PASO 4: Algoritmos Genéticos (Aplicación Conceptual/Simple) 🧬**

- **Concepto:** Usar AG para encontrar una "buena" solución a un problema de optimización. Aquí, lo ilustraremos con un ejemplo muy simplificado, ya que optimizar K-Means con AG es más complejo.
- **Ejemplo Simple:** Supongamos que queremos encontrar un subconjunto de 2 características de RFM (Recency, Frequency, MonetaryValue) que maximice una métrica de calidad del clustering (ej. Silueta) para un K fijo.
  - **Cromosoma:** Una lista binaria de longitud 3 (ej. [1, 0, 1] significa usar Recency y MonetaryValue).
  - **Fitness:** Coeficiente de Silueta usando K-Means con K=optimal_k (el K que elegimos antes) sobre las características seleccionadas.

```python
# PASO 4: Algoritmos Genéticos (Aplicación Conceptual/Simple)

if 'rfm_scaled' in locals() and not rfm_scaled.empty and 'optimal_k' in locals():
    print("\nIniciando ejemplo conceptual de Algoritmo Genético... 🧬")

    features_list = ['Recency', 'Frequency', 'MonetaryValue']
    data_for_ga = rfm_scaled[features_list] # Usamos los datos escalados

    # --- Definición del Algoritmo Genético Simple ---
    def generate_chromosome():
        # Genera un cromosoma binario de longitud 3
        # Al menos una característica debe ser seleccionada
        while True:
            chromosome = [random.randint(0, 1) for _ in range(len(features_list))]
            if sum(chromosome) > 0: # Asegurar que al menos una característica esté seleccionada
                return chromosome

    def calculate_fitness(chromosome, data, k):
        selected_features_indices = [i for i, bit in enumerate(chromosome) if bit == 1]
        if not selected_features_indices:
            return -1 # Penalizar si no hay características seleccionadas

        selected_feature_names = [features_list[i] for i in selected_features_indices]
        subset_data = data[selected_feature_names]

        if subset_data.shape[1] < 1: # Debe haber al menos una característica
             return -1
        if len(subset_data) < k: # No se puede hacer clustering si hay menos muestras que clusters
            return -1

        kmeans_ga = KMeans(n_clusters=k, random_state=42, n_init=10)
        try:
            labels = kmeans_ga.fit_predict(subset_data)
            if len(set(labels)) < 2: # Silueta requiere al menos 2 clusters formados
                return -1 # Penalizar si no se forman suficientes clusters
            score = silhouette_score(subset_data, labels)
            return score
        except ValueError:
            return -1 # En caso de error en K-Means o silueta

    population_size = 10
    num_generations = 5 # Pocas generaciones para un ejemplo rápido
    mutation_rate = 0.1

    # 1. Inicializar Población
    population = [generate_chromosome() for _ in range(population_size)]
    print(f"Población Inicial (primeros 3): {population[:3]}")

    best_chromosome_overall = None
    best_fitness_overall = -1

    history_best_fitness = []

    # 2. Bucle Evolutivo
    for generation in range(num_generations):
        # 2a. Calcular Fitness
        fitness_scores = [calculate_fitness(chromo, data_for_ga, optimal_k) for chromo in population]

        # Guardar el mejor de esta generación
        current_best_idx = np.argmax(fitness_scores)
        current_best_fitness = fitness_scores[current_best_idx]
        history_best_fitness.append(current_best_fitness)

        if current_best_fitness > best_fitness_overall:
            best_fitness_overall = current_best_fitness
            best_chromosome_overall = population[current_best_idx]

        print(f"Generación {generation+1}: Mejor Fitness = {current_best_fitness:.3f}, Mejor Cromosoma = {population[current_best_idx]}")

        if generation == num_generations -1: # No hacer selección/cruce/mutación en la última generación
            break

        # 2b. Selección (Torneo simple)
        new_population = []
        for _ in range(population_size):
            parent1_idx, parent2_idx = random.sample(range(population_size), 2)
            winner = population[parent1_idx] if fitness_scores[parent1_idx] > fitness_scores[parent2_idx] else population[parent2_idx]
            new_population.append(list(winner)) # list() para hacer copia

        # 2c. Cruce (Punto único)
        for i in range(0, population_size, 2):
            if i + 1 < population_size and random.random() < 0.7: # Tasa de cruce
                parent1 = new_population[i]
                parent2 = new_population[i+1]
                crossover_point = random.randint(1, len(features_list) - 1)
                child1 = parent1[:crossover_point] + parent2[crossover_point:]
                child2 = parent2[:crossover_point] + parent1[crossover_point:]
                # Asegurar que los hijos sean válidos (al menos una característica)
                if sum(child1) > 0: new_population[i] = child1
                if sum(child2) > 0: new_population[i+1] = child2


        # 2d. Mutación
        for i in range(population_size):
            for j in range(len(features_list)):
                if random.random() < mutation_rate:
                    new_population[i][j] = 1 - new_population[i][j] # Flip bit
            # Asegurar que el cromosoma mutado sea válido
            if sum(new_population[i]) == 0:
                new_population[i] = generate_chromosome() # Regenerar si es inválido

        population = new_population

    print(f"\nMejor cromosoma encontrado después de {num_generations} generaciones: {best_chromosome_overall}")
    print(f"Mejor fitness (Silueta): {best_fitness_overall:.3f}")
    if best_chromosome_overall:
        selected_features_final = [features_list[i] for i, bit in enumerate(best_chromosome_overall) if bit == 1]
        print(f"Características seleccionadas por AG: {selected_features_final}")

    # Graficar la mejora del fitness
    plt.figure(figsize=(8,5))
    plt.plot(range(1, num_generations + 1), history_best_fitness, marker='o')
    plt.title('Mejora del Fitness (Silueta) por Generación (AG)')
    plt.xlabel('Generación')
    plt.ylabel('Mejor Coeficiente de Silueta')
    plt.grid(True)
    plt.show()

    print("Ejemplo de Algoritmo Genético completado. 🎉")
else:
    print("Datos necesarios no disponibles, saltando Algoritmos Genéticos.")
```

**🤔 Pregunta**

- ¿Cómo se representa una "solución" (cromosoma) en este problema de AG?
- ¿Qué mide la función de "fitness"? ¿Por qué es importante?
- Observa el gráfico de mejora del fitness. ¿Parece que el AG está convergiendo hacia una buena solución?
- ¿Cómo podrías usar AG para optimizar el número de clusters K en K-Means en lugar de las características? (Pista: ¿Qué sería el cromosoma y qué sería el fitness?)

## **PASO 5: Analítica de Asociación (Market Basket Analysis) 🧺🛒**

- **Concepto:** Encontrar reglas de asociación del tipo "Si un cliente compra {Producto A}, también tiende a comprar {Producto B}".
- **Preparación de Datos:** Necesitamos transformar los datos a un formato transaccional (una fila por transacción, con productos como columnas binarias o una lista de productos por transacción). Usaremos el enfoque de lista de productos.

```python
# PASO 5: Analítica de Asociación

if not df_processed.empty:
    print("\nIniciando Analítica de Asociación... 🔗")

    # Preparar datos para Apriori: necesitamos una lista de items por transacción
    # Agruparemos por 'Invoice' y listaremos los 'Description' de los productos
    # Es importante usar 'Description' en lugar de 'StockCode' para que las reglas sean más interpretables
    # Podríamos necesitar limpiar 'Description' (ej. quitar espacios extra)
    df_processed['Description'] = df_processed['Description'].str.strip()

    # Crear la lista de items por factura (transacción)
    # Filtrar facturas con muy pocos items podría ser útil, pero aquí lo mantenemos simple
    basket_data = (df_processed.groupby(['Invoice', 'Description'])['Quantity']
                   .sum().unstack().reset_index().fillna(0)
                   .set_index('Invoice'))

    # Convertir cantidades a 1 (presencia) o 0 (ausencia)
    def encode_units(x):
        if x <= 0:
            return 0
        if x >= 1:
            return 1

    basket_encoded = basket_data.applymap(encode_units)
    # Eliminar columnas (productos) que no se compraron en ninguna transacción (si las hay)
    basket_encoded = basket_encoded.loc[:, (basket_encoded != 0).any(axis=0)]

    print(f"\nDataset para Apriori (primeras filas, {basket_encoded.shape[1]} productos como columnas):")
    print(basket_encoded.head())

    if basket_encoded.shape[0] > 0 and basket_encoded.shape[1] > 0 :
        # Aplicar Apriori para encontrar itemsets frecuentes
        # min_support es un umbral crítico. Un valor bajo puede generar demasiados itemsets.
        # Empezaremos con un valor relativamente alto para este dataset grande.
        # Si no se encuentran itemsets, se deberá bajar.
        # Por el tamaño del dataset, esta operación puede ser lenta.
        # Para un ejemplo más rápido, podríamos tomar una muestra del df_processed.
        # Ejemplo: df_sample = df_processed.sample(frac=0.1, random_state=42) y rehacer basket_data

        frequent_itemsets = apriori(basket_encoded, min_support=0.02, use_colnames=True) # min_support=0.02 significa que el itemset aparece en al menos el 2% de las transacciones
        print("\nItemsets Frecuentes (con soporte >= 0.02):")
        print(frequent_itemsets.sort_values('support', ascending=False).head())

        if not frequent_itemsets.empty:
            # Generar reglas de asociación
            # 'metric' puede ser "confidence", "lift", etc. 'min_threshold' es el valor mínimo para esa métrica.
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

            # Filtrar y ordenar reglas (ej. por lift y confianza)
            rules_filtered = rules[(rules['lift'] >= 2) & (rules['confidence'] >= 0.2)] # Reglas interesantes
            print("\nReglas de Asociación (Lift >= 2, Confianza >= 0.2):")
            print(rules_filtered.sort_values(['lift', 'confidence'], ascending=[False, False]).head(10))

            # Visualización (ej. scatter plot de Support vs Confidence coloreado por Lift)
            if not rules_filtered.empty:
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x="support", y="confidence", size="lift", hue="lift",
                                data=rules_filtered, palette="viridis", sizes=(20, 200))
                plt.title('Reglas de Asociación: Support vs Confidence (Tamaño/Color por Lift)')
                plt.xlabel('Support')
                plt.ylabel('Confidence')
                plt.grid(True)
                plt.show()
            else:
                print("No se encontraron reglas con los filtros actuales para graficar.")
        else:
            print("No se encontraron itemsets frecuentes con el min_support actual. Intenta bajarlo.")
    else:
        print("El basket_encoded está vacío o no tiene productos. No se puede ejecutar Apriori.")

    print("Analítica de Asociación completada! 🛍️➡️🎁")

else:
    print("Dataset procesado no disponible, saltando Analítica de Asociación.")
```

**🤔 Pregunta**

- ¿Qué significa support, confidence y lift en el contexto de las reglas de asociación?

- Interpreta una de las reglas generadas.

  Por ejemplo, si encuentras {ROSES REGENCY TEACUP AND SAUCER } -> {GREEN REGENCY TEACUP AND SAUCER}, ¿qué te dice?

- ¿Cómo podría la tienda usar estas reglas para aumentar las ventas o mejorar la experiencia del cliente? (Ej. promociones, disposición de productos en la web/tienda).

- ¿Qué pasaría si usas un min_support muy bajo? ¿Y uno muy alto?

## **PASO 6: Modelos Ocultos de Márkov (HMM) - Aplicación Conceptual 🚶‍♂️➡️❓➡️☀️**

- **Concepto:** Modelar secuencias donde hay estados subyacentes (ocultos) que generan observaciones.
- **Aplicación Conceptual al Dataset:**
  - Podríamos intentar modelar "estados de compra del cliente" (ej. "explorador", "comprador regular", "a punto de abandonar") basados en secuencias de sus compras (ej. tipo de producto comprado, monto gastado por visita).
  - Esto es avanzado y requiere una cuidadosa ingeniería de características secuenciales.
  - **Para esta práctica, haremos un ejemplo más simple y autocontenido para ilustrar HMM, no directamente sobre el dataset de retail para no complicar excesivamente la preparación de datos secuenciales complejos.**

```python
# PASO 6: Modelos Ocultos de Márkov (HMM) - Ejemplo Ilustrativo

print("\nIniciando ejemplo ilustrativo de Modelo Oculto de Márkov... 🎰")

# Ejemplo: Un casino con dos dados, uno normal (N) y uno trucado (T).
# No sabemos qué dado se usa (estado oculto), solo vemos el resultado del lanzamiento (observación).

# Estados Ocultos: 0 = Dado Normal, 1 = Dado Trucado
# Observaciones: 0=1, 1=2, 2=3, 3=4, 4=5, 5=6 (resultados del dado)

# Parámetros del HMM (inventados para el ejemplo)
# n_components = número de estados ocultos
model_hmm = hmm.MultinomialHMM(n_components=2, random_state=42, n_trials=10) # n_trials para evitar convergencia prematura

# Probabilidades iniciales de estado (pi)
model_hmm.startprob_ = np.array([0.7, 0.3]) # 70% de empezar con dado Normal, 30% con Trucado

# Matriz de transición de estado (A)
#       N    T
#   N [0.8, 0.2]  (Si estaba en Normal, 80% sigue Normal, 20% cambia a Trucado)
#   T [0.4, 0.6]  (Si estaba en Trucado, 40% cambia a Normal, 60% sigue Trucado)
model_hmm.transmat_ = np.array([[0.8, 0.2],
                                [0.4, 0.6]])

# Matriz de emisión/observación (B)
#               Resultado del dado (1, 2, 3, 4, 5, 6)
# Estado Normal [1/6, 1/6, 1/6, 1/6, 1/6, 1/6] (Dado justo)
# Estado Trucado[1/10, 1/10, 1/10, 1/10, 1/10, 5/10] (Más probable que salga 6)
model_hmm.emissionprob_ = np.array([[1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
                                   [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]])

# Generar una secuencia de observaciones de este modelo (simulación)
# X es la secuencia de observaciones (lanzamientos), Z es la secuencia de estados ocultos (qué dado se usó)
X_observed, Z_states = model_hmm.sample(100) # Generar 100 lanzamientos

print("\nSecuencia de Observaciones (lanzamientos) generada (primeros 20):")
print(X_observed.flatten()[:20]) # flatten() para convertir a 1D array
print("\nSecuencia de Estados Ocultos REAL (qué dado se usó) (primeros 20):")
print(Z_states[:20])

# Tarea 1: Decodificación (Algoritmo de Viterbi)
# Dada la secuencia X_observed, ¿cuál es la secuencia de estados más probable?
logprob, Z_predicted = model_hmm.decode(X_observed, algorithm="viterbi")
print("\nSecuencia de Estados Ocultos PREDICHA por Viterbi (primeros 20):")
print(Z_predicted[:20])

accuracy_viterbi = np.mean(Z_states == Z_predicted)
print(f"Precisión de Viterbi al predecir el estado oculto: {accuracy_viterbi:.2%}")

# Visualizar las secuencias de estados (real vs. predicha)
plt.figure(figsize=(15, 6))
plt.subplot(2, 1, 1)
plt.plot(Z_states[:50], 'bo-', label='Estado Real (0=Normal, 1=Trucado)')
plt.title('Estados Ocultos Reales vs. Predichos (Primeros 50 puntos)')
plt.ylabel('Estado')
plt.yticks([0,1], ['Normal', 'Trucado'])
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(Z_predicted[:50], 'ro-', label='Estado Predicho (Viterbi)')
plt.xlabel('Tiempo (Lanzamiento)')
plt.ylabel('Estado')
plt.yticks([0,1], ['Normal', 'Trucado'])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Tarea 2: Aprendizaje (Baum-Welch) - ¿Podemos aprender los parámetros del modelo a partir de X_observed?
# Supongamos que NO conocemos model_hmm.transmat_ ni model_hmm.emissionprob_
# Creamos un nuevo modelo HMM para entrenar
model_hmm_learned = hmm.MultinomialHMM(n_components=2, random_state=123, n_iter=100, tol=0.01, n_trials=10)
# Necesitamos darle las observaciones en el formato correcto (lista de arrays, cada array es una secuencia)
# Para este ejemplo, X_observed ya está casi en el formato correcto (necesita ser 2D array)
# model_hmm_learned.fit(X_observed) # X_observed debe ser [n_samples, n_features]
# Para secuencias múltiples, X sería una concatenación y lengths un array con las longitudes de cada secuencia
# Para una única secuencia larga:
model_hmm_learned.fit(X_observed)


print("\nParámetros APRENDIDOS por Baum-Welch:")
print("Probabilidades iniciales aprendidas:\n", model_hmm_learned.startprob_)
print("Matriz de transición aprendida:\n", model_hmm_learned.transmat_)
print("Matriz de emisión aprendida:\n", model_hmm_learned.emissionprob_)
print("\nComparar con los parámetros originales:")
print("Original startprob_:\n", model_hmm.startprob_)
print("Original transmat_:\n", model_hmm.transmat_)
print("Original emissionprob_:\n", model_hmm.emissionprob_)


print("Ejemplo de HMM completado. 🎲")
```

**🤔 Preguntas para el estudiante:**

- En el ejemplo del dado, ¿qué representan los estados ocultos y qué representan las observaciones?
- ¿Qué hace el algoritmo de Viterbi? ¿Cuán bien predijo la secuencia de dados usados?
- ¿Qué intenta hacer el algoritmo de Baum-Welch? Compara los parámetros aprendidos con los originales. ¿Se parecen? ¿Qué podría afectar la calidad del aprendizaje?
- Piensa en el dataset de e-commerce. ¿Qué podría ser una "secuencia de observaciones" para un cliente? ¿Qué "estados ocultos" del cliente podrían generar esas secuencias? (Ej. secuencia de categorías de productos comprados, estados: "buscando regalos", "compra semanal", "interés en electrónica").

## **PASO 7: Reflexión Final y "Eficiencia" de los Algoritmos 🧠**

- **Texto para el Estudiante (en Colab Markdown):**

  "¡Felicidades por completar esta extensa práctica! 🎉 Hemos aplicado varios algoritmos de aprendizaje no supervisado a un dataset de e-commerce (y un ejemplo ilustrativo para HMM).

  **Reflexión sobre la "Eficiencia" y Utilidad:**

  No tiene mucho sentido preguntar cuál de estos algoritmos es "más eficiente" en un sentido absoluto, ya que **sirven para propósitos muy diferentes**:

  

  1. **K-Means (Agrupamiento):**

     - **Utilidad:** Excelente para segmentar clientes en grupos distintos basados en su comportamiento (RFM). Permite personalizar estrategias.

     - **"Eficiencia":** Eficiente en encontrar estos grupos si los datos son adecuados (clusters globulares) y K se elige bien. Su "eficiencia" es la calidad de los segmentos y los insights que proporcionan.

       

  2. **PCA (Reducción de Dimensionalidad):**

     - **Utilidad:** Muy útil para reducir el número de características, facilitando la visualización (como vimos con RFM), combatiendo la maldición de la dimensionalidad y a veces mejorando el rendimiento de otros algoritmos al reducir el ruido.

     - **"Eficiencia":** Eficiente en capturar la máxima varianza con menos dimensiones. Su "eficiencia" radica en la compresión de información sin pérdida significativa (o con pérdida controlada).

       

  3. **Algoritmos Genéticos (Optimización):**

     - **Utilidad:** Son herramientas de búsqueda y optimización. Los usamos conceptualmente para la selección de características. Podrían usarse para optimizar hiperparámetros de K-Means (como K o la inicialización de centroides) o para problemas complejos de enrutamiento de entregas, etc.

     - **"Eficiencia":** Su "eficiencia" es encontrar soluciones buenas (o casi óptimas) en espacios de búsqueda grandes y complejos donde otros métodos fallan. Pueden ser computacionalmente intensivos.

       

  4. **Analítica de Asociación (Minería de Reglas):**

     - **Utilidad:** Perfecta para descubrir relaciones entre productos ("market basket analysis"). Genera reglas accionables para cross-selling, up-selling, diseño de promociones, etc.

     - **"Eficiencia":** Medida por la calidad (soporte, confianza, lift) e interpretabilidad de las reglas encontradas. Algoritmos como Apriori son eficientes para datasets de tamaño moderado.

       

  5. **Modelos Ocultos de Márkov (Modelado Secuencial):**

     - **Utilidad:** Ideales para datos donde el orden importa y hay estados subyacentes no observables (ej. reconocimiento de voz, bioinformática, y conceptualmente, modelar fases del ciclo de vida del cliente basadas en secuencias de acciones).

     - **"Eficiencia":** Su "eficiencia" se mide por qué tan bien el modelo aprendido representa las secuencias observadas y cuán precisas son sus predicciones de estados o futuras observaciones.

       

  ### **Conclusión:**

  Cada algoritmo es una herramienta poderosa en la caja del científico de datos. La clave es entender **qué problema resuelve cada uno** y **cuándo aplicarlo**. A menudo, estas técnicas se pueden usar de forma **complementaria**. Por ejemplo:

  

  - PCA para reducir dimensiones antes de K-Means.
  - Resultados de K-Means (segmentos) para analizar reglas de asociación específicas por segmento.
  - 

  ¡Sigue explorando y experimentando! El aprendizaje no supervisado es un campo vasto y fascinante para descubrir patrones ocultos en tus datos. 🚀"