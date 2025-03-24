# ProyectoFinal-4Geeks
# Clasificación de Abejas: Apis vs Bombus con EfficientNetB0

Este proyecto tiene como objetivo clasificar imágenes de dos tipos de abejas —**Apis** y **Bombus**— utilizando modelos de deep learning con **TensorFlow/Keras**. Se emplea **EfficientNetB0** como base para transfer learning, junto con técnicas de limpieza, balanceo de datos, data augmentation y ajuste de threshold.

---

## Estructura del Proyecto
├── data/
│ ├── datos1.csv
│ ├── datos2.csv
│ ├── datos3.csv
│ ├── datos4.csv
│ ├── datos5.csv
│ ├── images_apis/ # Imágenes descargadas de Apis │ 
| └── images_bombus/ # Imágenes descargadas de Bombus │ 
├── dataset_final/
│ └── train/
│   ├── 0/ # Imágenes de Apis (clase 0) 
│   └── 1/ # Imágenes de Bombus (clase 1) │ 
├── augmented/
│   ├── augmented_apis/ # Imágenes aumentadas de Apis 
│   └── augmented_bombus/ # Imágenes aumentadas de Bombus 
├── modelo_abejas.keras # Modelo entrenado sin transfer learning 
├── modelo_abejas_transfer.keras # Modelo afinado con transfer learning │ 
├── PROYECTO_FINAL.ipynb # Notebook principal del proyecto 
├── README.md # Explicación y documentación del proyecto 
└── requirements.txt # Librerías necesarias

---

## Objetivo

Desarrollar un modelo robusto capaz de distinguir entre **Apis** y **Bombus** a partir de imágenes reales. Esto puede ser aplicado a tareas de biodiversidad, monitoreo ambiental y estudios entomológicos y apicultores.

---

## Flujo de Trabajo

### 1. Carga y limpieza de datos
- Se combinan 5 archivos CSV en un único DataFrame.
- Se filtran especies que contengan `apis` o `bombus`.
- Se eliminan duplicados y se convierten los nombres a minúsculas.
- Se identifican imágenes corruptas y se eliminan físicamente del disco.

### 2. Descarga y organización de imágenes
- Las imágenes se descargan desde URLs y se guardan en carpetas por clase (`apis` y `bombus`).
- Se crea un nuevo DataFrame con rutas válidas y etiquetas binarias (`0 = apis`, `1 = bombus`).

### 3. División del dataset
- Se divide en `train`, `validation` y `test` (60/20/20), usando estratificación para mantener proporciones de clases.
- Se asegura que no haya solapamientos entre los subconjuntos.

### 4. Preparación de datos con `tf.data.Dataset`
- Se crean datasets eficientes a partir de rutas de imagen.
- Las imágenes se preprocesan usando `preprocess_input` de EfficientNet.

### 5. Entrenamiento sin Data Augmentation
- Se entrena un modelo basado en EfficientNetB0 congelado preentrenado con Imagenet.
- Se emplea `EarlyStopping` y `class_weight` para evitar sobreajuste y compensar el desbalance.
- Se evalúa el modelo en test y se ajusta el **threshold óptimo** para maximizar el F1-score.

### 6. Data Augmentation
- Se aplica `ImageDataGenerator` para generar imágenes artificiales por clase.
- Se guarda el output en carpetas separadas por clase.
- Se combina todo en un nuevo `DataFrame` balanceado.

### 7. Transfer Learning (fine-tuning)
- Se cargan los pesos del modelo anterior y se reentrenan las últimas capas.
- Se entrena sobre el dataset aumentado.
- Se evalúa nuevamente en validación y test.

### 8. Evaluación del modelo
- Se grafican curvas:
  - Accuracy / Loss
  - Precision / Recall
  - F1-score vs Threshold
  - ROC
- Se generan matrices de confusión para `train`, `val` y `test`.
- Se calcula el **threshold óptimo** para producción.

### 9. Inferencia y pruebas
- Se selecciona una imagen aleatoria y se clasifica con ambos modelos (con y sin Transfer Learning).
- Se visualiza el resultado junto con la probabilidad y clase predicha.

---

## Métricas destacadas

| Modelo                | F1-score máx. | Threshold óptimo  |
|---------------------- |---------------|------------------ |
| Sin Transfer Learning | 0.9531        | 0.4612            |
| Con Transfer Learning | 0.9549        | 0.42              |

Ambos modelos muestran alta precisión, pero el uso de Transfer Learning + Augmentation mejora ligeramente la generalización.

---

## Decisiones técnicas destacadas

- **Uso de carpetas en lugar de arrays en DataFrame para augmentation:** evita errores de memoria y facilita reutilización.
- **Eliminación de imágenes corruptas:** mejora la calidad del entrenamiento.
- **Ajuste de threshold personalizado:** permite adaptar el modelo a necesidades específicas (recall vs precisión).

---

## Requisitos

- Python 3.8+
- TensorFlow 2.10+
- pandas, numpy, scikit-learn, matplotlib, seaborn
- PIL (Pillow)
- tqdm

Instalación:

```bash
conda install -n nombre_entorno python=3.10
pip install -r requirements.txt
