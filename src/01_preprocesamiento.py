import argparse
import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image
import io

EXTENSIONES_IMAGEN = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
# 0 = Real, 1 = Fake/IA (según carpetas REAL / FAKE)
CLASES = {'REAL': 0, 'FAKE': 1}
SPLITS = ('train', 'test')
COLUMNAS = [
    'Varianza_Laplaciana', 'Entropia_Color',
    'Media_B', 'Media_G', 'Media_R',
    'Std_B', 'Std_G', 'Std_R',
    'Energia_Histograma', 'ELA', 'Target'
]


def extraer_caracteristicas(ruta_imagen):
    img_bgr = cv2.imread(ruta_imagen)
    if img_bgr is None:
        return None

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Varianza laplaciana
    varianza_bordes = cv2.Laplacian(img_gray, cv2.CV_64F).var()

    # Entropía de color
    entropias = []
    for canal in cv2.split(img_bgr):
        hist = cv2.calcHist([canal], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        entropias.append(-np.sum(hist * np.log2(hist)))
    entropia_color = np.mean(entropias)

    # Media y desviación por canal
    medias = [np.mean(c) for c in cv2.split(img_bgr)]
    stds = [np.std(c) for c in cv2.split(img_bgr)]

    # Energía del histograma
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    energia = np.sum(hist**2)

    # ELA
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)
    img_comprimida = Image.open(buffer)
    ela = np.abs(np.array(img_pil).astype(float) - np.array(img_comprimida).astype(float)).mean()

    return [varianza_bordes, entropia_color,
            medias[0], medias[1], medias[2],
            stds[0], stds[1], stds[2],
            energia, ela]


def _es_imagen(nombre_archivo):
    return nombre_archivo.lower().endswith(EXTENSIONES_IMAGEN)


def procesar_split(ruta_base, split, max_por_clase=None):
    """Procesa dataset_raw/{split}/{REAL|FAKE}/."""
    datos = []

    for clase, etiqueta in CLASES.items():
        ruta_carpeta = os.path.join(ruta_base, split, clase)

        if not os.path.exists(ruta_carpeta):
            print(f"AVISO: No se encontro la carpeta: {ruta_carpeta}")
            continue

        print(f"Procesando {split}/{clase}...")
        procesadas_clase = 0

        for root, _, files in os.walk(ruta_carpeta):
            for archivo in files:
                if max_por_clase is not None and procesadas_clase >= max_por_clase:
                    break
                if not _es_imagen(archivo):
                    continue

                ruta_img = os.path.join(root, archivo)
                features = extraer_caracteristicas(ruta_img)

                if features is not None:
                    datos.append(features + [etiqueta])
                    procesadas_clase += 1

                    if procesadas_clase % 1000 == 0:
                        print(f"  ... {procesadas_clase} imágenes en {split}/{clase}")

            if max_por_clase is not None and procesadas_clase >= max_por_clase:
                break

        print(f"  OK {split}/{clase}: {procesadas_clase} imagenes")

    return datos


def guardar_csv(datos, ruta_salida):
    df = pd.DataFrame(datos, columns=COLUMNAS)
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    df.to_csv(ruta_salida, index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extrae patrones numéricos del dataset en dataset_raw/"
    )
    parser.add_argument(
        "--ruta-dataset",
        default="dataset_raw/",
        help="Carpeta raíz con subcarpetas train/ y test/",
    )
    parser.add_argument(
        "--max-por-clase",
        type=int,
        default=None,
        help="Límite de imágenes por clase y split (útil para pruebas rápidas)",
    )
    args = parser.parse_args()

    print("Iniciando extracción de patrones...")
    print(f"Estructura esperada: {{ruta}}/train|test/{{REAL|FAKE}}/")
    if args.max_por_clase:
        print(f"Modo limitado: máximo {args.max_por_clase} imágenes por clase y split")

    alguno_procesado = False

    for split in SPLITS:
        dataset_split = procesar_split(
            args.ruta_dataset, split, max_por_clase=args.max_por_clase
        )

        if not dataset_split:
            print(f"AVISO: Sin imagenes procesadas para split '{split}'")
            continue

        ruta_salida = f"data/patrones_{split}.csv"
        df = guardar_csv(dataset_split, ruta_salida)
        print(f"CSV generado: {ruta_salida} ({len(df)} patrones)")
        alguno_procesado = True

    if not alguno_procesado:
        print("ERROR: No se procesaron imagenes. Revisa la ruta y la estructura del dataset.")
    else:
        print("Listo. Usá patrones_train.csv para entrenar y patrones_test.csv para evaluar.")
