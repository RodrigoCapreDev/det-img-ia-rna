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
2. Instalar dependencias: 
   ```bash
   pip install -r requirements.txt
   ```
3. Obtener el Dataset (Elegir una de las dos opciones):
    - Opción A (Automática mediante API de Kaggle):

      Entrá a tu cuenta en [Kaggle](https://www.kaggle.com/settings/), vas a tus configuraciones de perfil (Settings) y hacé clic en Create New Token. 
      
      Crear un archivo llamado `kaggle.json` en la raiz del repositorio (ver `kaggle.json.example`), copiar token y username.

      Abrí y ejecutá la notebook `src/00_descargar_dataset.ipynb` para descargar y estructurar las imágenes automáticamente en la carpeta `dataset_raw/`.

    - Opción B (Manual):

        Descargá el dataset directamente desde [Kaggle: AI vs Real Images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images).

        Descomprimilo manualmente dentro de la carpeta `dataset_raw/` asegurándote de respetar la siguiente estructura:
        ```text
        det-img-ia-rna/       <-- (Raíz del repositorio)
        ├── src/
        ├── data/
        └── dataset_raw/
            ├── train/
            │   ├── REAL/
            │   └── FAKE/
            └── test/
                ├── REAL/
                └── FAKE/
        ```
4. Ejecutar el script de preprocesamiento (genera un CSV por split):
    ```bash
    python src/preprocesamiento.py
    ```
    El dataset completo tiene ~120.000 imágenes; para una prueba rápida:
    ```bash
    python src/preprocesamiento.py --max-por-clase 100
    ```