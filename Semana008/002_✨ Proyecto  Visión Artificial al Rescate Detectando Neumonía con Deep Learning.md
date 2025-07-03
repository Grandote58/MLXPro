# **✨ Proyecto | Visión Artificial al Rescate: Detectando Neumonía con Deep Learning**

## 🧠 ¡Hola de nuevo, Data Scientist con visión de futuro!  🧠 

En nuestro proyecto anterior, enseñamos a una máquina a "leer" datos tabulares. Hoy, daremos un salto cuántico: le enseñaremos a **"ver"**. Nos adentraremos en el fascinante mundo de las Redes Neuronales Convolucionales (CNN) para resolver un problema diagnóstico crítico.

## 🚀 El Reto: Hospital "Valle de la Salud" en Crisis

Imagina que eres el/la líder del nuevo equipo de IA del Hospital "Valle de la Salud". Durante los picos estacionales de enfermedades respiratorias, el departamento de radiología se ve abrumado. Los radiólogos, aunque expertos, se enfrentan a cientos de radiografías de tórax al día, aumentando el riesgo de fatiga y retrasos en los diagnósticos.

La dirección del hospital te plantea un reto:

> *"No buscamos reemplazar a nuestros radiólogos, sino darles un 'asistente de IA' superdotado. Necesitamos un modelo que pueda analizar una radiografía de tórax y realizar un primer triaje, identificando con alta probabilidad los casos de neumonía. Esto permitiría a nuestro personal priorizar los casos más urgentes. *
>
> *La precisión global es importante, pero es **absolutamente crítico que no se nos escape ningún caso positivo (minimizar los Falsos Negativos)**."*

Tu misión es liderar el desarrollo de este sistema de visión artificial, desde la exploración de las imágenes hasta la evaluación de un modelo de Deep Learning, explicando su rendimiento en un lenguaje que un médico pueda entender y confiar.

## 🎯 Objetivos de la Práctica

Al completar este desafío, habrás dominado las siguientes competencias:

1. **Gestionar y preparar un dataset de imágenes** para un modelo de Deep Learning.
2. **Aplicar técnicas de aumento de datos (Data Augmentation)** para crear un modelo más robusto y generalizable.
3. **Diseñar, justificar y construir una Red Neuronal Convolucional (CNN)** desde cero utilizando TensorFlow y Keras.
4. **Entrenar un modelo de Deep Learning**, monitorizando su rendimiento durante el proceso.
5. **Evaluar la eficiencia del modelo de visión artificial** con métricas clínicas clave, enfocándose en la matriz de confusión.
6. **Interpretar los resultados visuales y numéricos** para proponer un plan de implementación en un entorno real.

## 🌐 El Dataset: Radiografías de Tórax (Neumonía)

Utilizaremos un dataset público muy popular de Kaggle, que contiene miles de imágenes de radiografías de tórax, ya clasificadas por pediatras en dos categorías: Normal y Neumonía.

**URL del Dataset en Kaggle:** `https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia`

## 💻 Desarrollo del Proyecto en Google Colab

¡Prepara tu entorno de Colab, que empieza el viaje al interior de las imágenes!

### 📘 Nota Didáctica: Fase 1 - Configurando Nuestro Laboratorio de Visión

Trabajar con datasets de imágenes grandes requiere una configuración inicial. En lugar de subir los datos manualmente (lo que sería muy lento), nos conectaremos directamente a Kaggle usando su API. ¡Esta es una habilidad fundamental para cualquier Data Scientist!

#### **Paso 1.1: Conectar Colab con Kaggle**

```python
# 🧠 Paso 1: Instalar la librería de Kaggle.
!pip install kaggle

# 🧠 Paso 2: Crear una carpeta para guardar tu token de API de Kaggle.
# Para que esto funcione, primero debes ir a tu perfil de Kaggle -> 'Account' -> 'API' -> 'Create New API Token'.
# Esto descargará un archivo 'kaggle.json'. Súbelo a tu entorno de Colab usando el panel de la izquierda.
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# 🧠 Paso 3: Descargar el dataset de radiografías de tórax.
# El comando es 'kaggle datasets download -d <nombre-del-dataset-en-kaggle>'
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

# 🧠 Paso 4: Descomprimir el archivo descargado.
# El '-q' es para que no muestre todos los nombres de archivo al descomprimir (¡son miles!).
!unzip -q chest-xray-pneumonia.zip
```

#### **Paso 1.2: Explorando la Estructura de Nuestros Datos Visuales**

```python
# 🧠 Importamos las librerías necesarias.
# TensorFlow y Keras son nuestro motor de Deep Learning.
# Matplotlib y OpenCV nos ayudarán a visualizar y manipular las imágenes.

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 # OpenCV para el manejo de imágenes
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# 🧠 Definimos las rutas a nuestras carpetas de datos.
# El dataset ya viene convenientemente dividido en entrenamiento, validación y prueba.
train_dir = 'chest_xray/train'
val_dir = 'chest_xray/val'
test_dir = 'chest_xray/test'

# 🧠 Veamos algunas imágenes de ejemplo para entender cómo son.
def plot_sample_images(directory, class_name, num_samples=3):
    class_dir = os.path.join(directory, class_name)
    sample_images = np.random.choice(os.listdir(class_dir), num_samples)

    plt.figure(figsize=(12, 4))
    for i, img_name in enumerate(sample_images):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path)
        plt.subplot(1, num_samples, i+1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f'{class_name}')
        plt.axis('off')
    plt.show()

print("Ejemplos de Radiografías Normales:")
plot_sample_images(train_dir, 'NORMAL')

print("\nEjemplos de Radiografías con Neumonía:")
plot_sample_images(train_dir, 'PNEUMONIA')
```

### 📘 Nota Didáctica: Fase 2 - Preparando las Imágenes para el "Ojo" de la IA

Una red neuronal no "ve" una imagen como nosotros. La ve como una matriz de números (píxeles). Nuestra tarea en el **preprocesamiento** es estandarizar estas matrices y, lo más importante, usar **aumento de datos (Data Augmentation)**.

> **¿Qué es el Data Augmentation?** Es una técnica para crear versiones modificadas de nuestras imágenes de entrenamiento (rotadas, con zoom, invertidas, etc.). 
>
> Esto enseña al modelo a reconocer una neumonía sin importar si la radiografía está ligeramente girada o si el paciente estaba más cerca o más lejos. ¡Es la clave para evitar que el modelo "memorice" las imágenes y aprenda a generalizar!

#### **Paso 2.1: Creando Generadores de Imágenes**

```python
# 🧠 Definimos las dimensiones a las que redimensionaremos todas las imágenes y el tamaño del lote (batch size).
IMG_WIDTH, IMG_HEIGHT = 150, 150
BATCH_SIZE = 32

# 🧠 Creamos el generador para los datos de entrenamiento CON AUMENTO DE DATOS.
# Rescalamos los píxeles (de 0-255 a 0-1), aplicamos rotaciones, zoom, etc.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 🧠 Para los datos de validación y prueba, SOLO re-escalamos.
# ¡Nunca aumentamos los datos con los que evaluamos, pues deben reflejar la realidad!
test_val_datagen = ImageDataGenerator(rescale=1./255)

# 🧠 Creamos los generadores que leerán las imágenes desde las carpetas.
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary' # Porque es una clasificación de 2 clases (Normal/Pneumonia)
)

validation_generator = test_val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False # Importante para la evaluación
)

test_generator = test_val_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False # Importante para la evaluación
)
```

### 📘 Nota Didáctica: Fase 3 - Diseñando la Arquitectura de Nuestra Red Neuronal

Ahora, construiremos el "cerebro" de nuestro sistema. Una **Red Neuronal Convolucional (CNN)** es perfecta para imágenes porque imita vagamente cómo la corteza visual humana procesa la información:

1. **Capas Convolucionales (`Conv2D`):** Actúan como "detectores de características". Las primeras capas aprenden a ver bordes y texturas simples. Las más profundas aprenden a combinar esas características para ver formas complejas como costillas o patrones pulmonares.
2. **Capas de Agrupación (`MaxPooling2D`):** Reducen el tamaño de la imagen para hacer el modelo más eficiente y ayudan a que sea robusto a pequeñas traslaciones.
3. **Capas Densas (`Dense`):** Son las capas de clasificación final que toman las características aprendidas y deciden si la imagen corresponde a "Normal" o "Neumonía".

#### **Paso 3.1: Construcción del Modelo CNN**

```python
# 🧠 Construimos el modelo capa por capa usando la API Secuencial de Keras.

model = Sequential([
    # 1er Bloque Convolucional
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    # 2do Bloque Convolucional
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    # 3er Bloque Convolucional
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    # Aplanar los mapas de características para las capas densas
    Flatten(),

    # Capas Densas para la clasificación
    Dense(512, activation='relu'),
    Dropout(0.5), # Dropout para prevenir el sobreajuste

    # Capa de Salida
    Dense(1, activation='sigmoid') # Sigmoid para clasificación binaria
])

# 🧠 Compilamos el modelo.
# 'adam' es un optimizador eficiente.
# 'binary_crossentropy' es la función de pérdida correcta para clasificación binaria.
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 🧠 Vemos un resumen de nuestra arquitectura.
model.summary()
```

#### **Paso 3.2: Entrenamiento del Modelo**

```python
# 🧠 Definimos un callback de 'EarlyStopping' para que el entrenamiento se detenga
# si el rendimiento en el set de validación no mejora después de varias épocas.
# Esto nos ahorra tiempo y evita el sobreajuste.
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 🧠 ¡A entrenar! Este proceso puede tardar varios minutos.
# Colab usará una GPU si está disponible, lo que acelera enormemente el proceso.
history = model.fit(
    train_generator,
    epochs=25, # Un número máximo de épocas
    validation_data=validation_generator,
    callbacks=[early_stopping]
)
```

### 📘 Nota Didáctica: Fase 4 - El Diagnóstico Final: Evaluación Crítica

El modelo ha sido entrenado. Ahora, como buenos científicos, debemos evaluar su rendimiento de forma objetiva en el set de prueba, que el modelo nunca ha visto. Aquí es donde respondemos a la pregunta del hospital: ¿podemos confiar en este asistente de IA?

#### **Paso 4.1: Visualizando el Aprendizaje**

```python
# 📊 Graficamos la precisión y la pérdida durante el entrenamiento.
# Esto nos dice si el modelo aprendió correctamente o si hay problemas como el sobreajuste.

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_ran = range(len(acc))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_ran, acc, 'b', label='Precisión de Entrenamiento')
plt.plot(epochs_ran, val_acc, 'r', label='Precisión de Validación')
plt.title('Precisión de Entrenamiento y Validación')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_ran, loss, 'b', label='Pérdida de Entrenamiento')
plt.plot(epochs_ran, val_loss, 'r', label='Pérdida de Validación')
plt.title('Pérdida de Entrenamiento y Validación')
plt.legend()

plt.show()
```

#### **Paso 4.2: Evaluación Final con la Matriz de Confusión**

```python
# 🧠 Evaluamos el modelo con el generador de prueba.
loss, accuracy = model.evaluate(test_generator)
print(f"✅ Accuracy en el set de prueba: {accuracy:.2%}")
print(f"📉 Pérdida en el set de prueba: {loss:.4f}")
print("-" * 50)

# 🧠 Obtenemos las predicciones para el set de prueba.
y_pred_prob = model.predict(test_generator)
y_pred = (y_pred_prob > 0.5).astype("int32")
y_true = test_generator.classes

# 🧠 Imprimimos el reporte de clasificación y la matriz de confusión.
from sklearn.metrics import classification_report, confusion_matrix

print("📊 Reporte de Clasificación:")
# test_generator.class_indices nos da el mapeo de 'NORMAL': 0, 'PNEUMONIA': 1
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))
print("-" * 50)

print("🔍 Matriz de Confusión:")
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=test_generator.class_indices.keys(),
            yticklabels=test_generator.class_indices.keys())
plt.title('Matriz de Confusión del Set de Prueba')
plt.ylabel('Etiqueta Real')
plt.xlabel('Etiqueta Predicha')
plt.show()
```

> **Conclusión para el Hospital "Valle de la Salud":** 
>
> "Equipo, nuestro primer prototipo de IA ha logrado una precisión general del 88% en datos nunca antes vistos. Más importante aún, al analizar la Matriz de Confusión, vemos que de los 390 casos reales de neumonía, nuestro modelo identificó correctamente 369. 
>
> Esto nos da un **Recall (sensibilidad) del 95% para los casos de neumonía**. Esto significa que el modelo es extremadamente bueno para 'levantar la bandera' cuando hay una posible neumonía. 
>
> *Tuvimos 21 Falsos Negativos (casos que se nos escaparon), que es el área clave a mejorar. Sin embargo, como primer asistente de triaje, este rendimiento es muy prometedor para ayudar a nuestros radiólogos a priorizar su carga de trabajo."*

## ✅ ¿Qué has aprendido en esta misión de visión artificial?

¡Misión cumplida! Has construido un sistema de IA capaz de interpretar imágenes médicas. 

Este es un gran paso en tu carrera como Data Scientist. Específicamente, has aprendido a:

- **Gestionar un Flujo de Trabajo de Deep Learning:** Desde la descarga de datos con una API hasta la evaluación final, has completado un proyecto de visión por computadora de principio a fin.
- **Dominar el Preprocesamiento de Imágenes:** Entendiste el rol crítico de la normalización y el **aumento de datos (Data Augmentation)**, una técnica indispensable para cualquier proyecto de visión.
- **Construir y Entender una CNN:** Ya no es una caja negra. Sabes qué son las capas `Conv2D` y `MaxPooling2D` y por qué son la base del reconocimiento de imágenes moderno.
- **Evaluar con Contexto Clínico:** Reforzaste la idea de que la `accuracy` no lo es todo. Aprendiste a priorizar el **Recall** (sensibilidad) en un problema médico donde los Falsos Negativos son el mayor riesgo.
- **Interpretar Gráficos de Entrenamiento:** Sabes cómo leer las curvas de pérdida y precisión para diagnosticar la salud de tu modelo durante el entrenamiento.
- **Comunicar Resultados Complejos:** Eres capaz de traducir una matriz de confusión en una recomendación estratégica y comprensible para un stakeholder no técnico, como un director de hospital.