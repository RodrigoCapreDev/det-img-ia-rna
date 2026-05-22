# Detección de Imágenes IA vs Reales (`det-img-ia-rna`)

Trabajo Práctico para la cátedra de Inteligencia Artificial - UTN FRLP.

## 📌 Descripción del Proyecto
Este sistema utiliza una red neuronal con aprendizaje **Backpropagation** (Perceptrón Multicapa) para clasificar imágenes sintéticas generadas por IA frente a fotografías reales. 

Debido a que una red densa no puede procesar eficientemente píxeles crudos, el proyecto aplica un preprocesamiento externo mediante filtros de convolución analíticos (**OpenCV**) para extraer métricas de textura, nitidez y ruido, consolidando los patrones en un archivo CSV liviano para el entrenamiento.

## 📁 Estructura del Repositorio
* `dataset_raw/` - Imágenes originales extraídas de Kaggle (No se suben al repo).
* `src/` - Scripts de preprocesamiento (`.py`) y Notebooks de entrenamiento (`.ipynb`).
* `data/` - CSVs con patrones (`patrones_train.csv`, `patrones_test.csv`).

## 🚀 Cómo empezar (Local)
1. Clonar el repositorio.
2. Descargar el dataset y descomprimirlo en `dataset_raw/` con esta estructura:
   ```
   dataset_raw/
   ├── train/
   │   ├── REAL/
   │   └── FAKE/
   └── test/
       ├── REAL/
       └── FAKE/
   ```
3. Instalar dependencias: 
   ```bash
   pip install -r requirements.txt
4. Ejecutar el script de preprocesamiento (genera un CSV por split):
    ```bash
    python src/preprocesamiento.py
    ```
    El dataset completo tiene ~120.000 imágenes; para una prueba rápida:
    ```bash
    python src/preprocesamiento.py --max-por-clase 100
    ```