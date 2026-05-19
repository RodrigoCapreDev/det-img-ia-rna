import cv2
import numpy as np
import pandas as pd
import os

def extraer_caracteristicas(ruta_imagen):
    """Aplica convolución y extrae features de una imagen."""
    # Leer imagen en escala de grises
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return None
        
    # Filtro 1: Laplaciano (Detecta bordes y texturas finas)
    laplaciano = cv2.Laplacian(img, cv2.CV_64F)
    varianza_bordes = laplaciano.var()
    
    # Filtro 2: Estadísticas de ruido/brillo
    brillo_medio = np.mean(img)
    desvio_brillo = np.std(img)
    
    return [varianza_bordes, brillo_medio, desvio_brillo]

def procesar_dataset(ruta_base):
    datos = []
    
    # Mapeo de carpetas a etiquetas (0 = Real, 1 = Fake/IA)
    categorias = {'RealArt': 0, 'AiArtData': 1}
    
    for carpeta, etiqueta in categorias.items():
        ruta_carpeta = os.path.join(ruta_base, carpeta)
        
        # Verificamos que la carpeta exista antes de iterar
        if not os.path.exists(ruta_carpeta):
            print(f"⚠️ No se encontró la carpeta: {ruta_carpeta}")
            continue
            
        print(f"Procesando carpeta: {carpeta}...")

        # Recorremos recursivamente y procesamos solo archivos de imagen
        for root, _, files in os.walk(ruta_carpeta):
            for archivo in files:
                # Filtrar por extensiones comunes de imagen
                if not archivo.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
                    continue

                ruta_img = os.path.join(root, archivo)

                # Extraer features
                features = extraer_caracteristicas(ruta_img)

                if features is not None:
                    # Agregamos los features y la salida esperada (Target)
                    fila = features + [etiqueta]
                    datos.append(fila)
                
    return datos

if __name__ == "__main__":
    # Asegurate de que esta ruta apunte a donde descargaste el dataset
    RUTA_DATASET = "../dataset_raw/"
    
    print("Iniciando extracción de patrones...")
    dataset_procesado = procesar_dataset(RUTA_DATASET)
    
    if dataset_procesado:
        # Armamos el CSV con Pandas
        columnas = ['Varianza_Laplaciana', 'Brillo_Medio', 'Desvio_Brillo', 'Target']
        df = pd.DataFrame(dataset_procesado, columns=columnas)
        
        # Creamos la carpeta data si no existe
        os.makedirs("../data", exist_ok=True)
        
        # Guardamos el archivo
        ruta_salida = "../data/patrones.csv"
        df.to_csv(ruta_salida, index=False)
        print(f"✅ ¡Éxito! CSV generado en: {ruta_salida} con {len(df)} patrones.")
    else:
        print("❌ No se procesaron imágenes. Revisá la ruta del dataset.")