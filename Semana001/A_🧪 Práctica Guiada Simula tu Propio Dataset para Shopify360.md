# **🧪 Práctica Guiada: "Simula tu Propio Dataset para Shopify360"**

## 📅 Objetivo

Crear un dataset simulado de clientes de eCommerce que contenga problemas comunes de calidad de datos (valores nulos, outliers, errores tipográficos, duplicados) para utilizar en tareas de preprocesamiento y modelado.

## 🛠 Paso a Paso en Google Colab

### 1. 🔄 Importar librerías

```python
import pandas as pd
import numpy as np
import random
```

### 2. 📊 Definir funciones auxiliares para errores intencionales

```python
def introducir_errores_genero(genero):
    variantes = {'masculino': ['Masculino', 'masculino', 'masculnio', 'MAScULINO'],
                 'femenino': ['Femenino', 'femenino', 'FEMENIO', ' FEMENINO ']}
    return random.choice(variantes[genero])

def introducir_valores_especiales_ingreso(valor):
    return random.choice([valor, -999, np.nan])

def introducir_clase_defectuosa(valor):
    return random.choice([valor, valor.lower(), '¿si?', 'sí'])
```

### 3. 📊 Crear datos simulados con errores

```python
np.random.seed(42)
n = 100

edades = np.random.normal(35, 10, n).astype(int)
edades[np.random.choice(n, 5, replace=False)] = np.nan  # Valores nulos

ingresos = np.random.normal(50000, 12000, n).astype(int)
ingresos = [introducir_valores_especiales_ingreso(x) for x in ingresos]

generos = [introducir_errores_genero(random.choice(['masculino', 'femenino'])) for _ in range(n)]
compras = [introducir_clase_defectuosa(random.choice(['Si', 'No'])) for _ in range(n)]

clientes = pd.DataFrame({
    'id_cliente': list(range(1000, 1000 + n)),
    'edad': edades,
    'genero': generos,
    'ingreso_mensual': ingresos,
    'compro': compras
})
```

### 4. 🤡 Introducir duplicados y guardar

```python
# Duplicar 5 registros
clientes = pd.concat([clientes, clientes.sample(5)], ignore_index=True)

# Guardar
clientes.to_csv("clientes_simulado_shopify360.csv", index=False)
from google.colab import files
files.download("clientes_simulado_shopify360.csv")
```

## 🔮 Resultado Esperado

El dataset generado incluirá:

- Errores ortográficos en "género".
- Valores especiales como `-999` o `NaN` en ingreso.
- Clases mal etiquetadas como "¿si?", "sí".
- Duplicados reales.
- Edad faltante (valores nulos).