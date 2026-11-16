"""
Preparacion de datos (CPU): replica el muestreo y split del Laboratorio 3
(Hugo2049/Laboratorio-3-data-science) para que los resultados de este
proyecto sean comparables con los del laboratorio original.

Esta fase se mantiene en CPU a proposito (lectura de disco + decodificacion
JPEG + resize con Pillow): no es una carga aritmetica masiva y paralelizable
en GPU, es I/O y manipulacion de archivos independiente por imagen, por lo
que el paralelismo que tiene sentido aqui es multi-nucleo de CPU, no CUDA.
"""

import os

# Cada worker solo decodifica/redimensiona UNA imagen a la vez, no necesita
# multihilo interno. Sin esto, NumPy/OpenBLAS lanza por defecto tantos hilos
# BLAS como nucleos tiene la maquina (aqui 32) DENTRO DE CADA proceso worker,
# lo que con varios workers agota la memoria/commit charge del sistema antes
# de que el multiprocessing llegue a ayudar. Debe fijarse ANTES de "import
# numpy" (aqui y en los procesos hijos, que reimportan este modulo).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import random
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

IMAGE_SIZE = 64
MUESTRAS_POR_CLASE = 600
SEED = 42


def listar_rutas_y_etiquetas(train_dir, muestras_por_clase=MUESTRAS_POR_CLASE, seed=SEED):
    random.seed(seed)
    clases = sorted(os.listdir(train_dir))

    rutas, etiquetas = [], []
    for letra in clases:
        carpeta = os.path.join(train_dir, letra)
        archivos = os.listdir(carpeta)
        seleccion = random.sample(archivos, muestras_por_clase)
        for archivo in seleccion:
            rutas.append(os.path.join(carpeta, archivo))
            etiquetas.append(letra)

    return clases, pd.DataFrame({"ruta": rutas, "etiqueta": etiquetas})


def dividir_train_val_test(datos, seed=SEED):
    train_df, temp_df = train_test_split(
        datos, test_size=0.3, stratify=datos["etiqueta"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["etiqueta"], random_state=seed
    )
    return train_df, val_df, test_df


def _cargar_una_imagen(ruta, tamano=IMAGE_SIZE):
    imagen = Image.open(ruta).convert("RGB").resize((tamano, tamano))
    return np.asarray(imagen, dtype=np.float32) / 255.0


def cargar_y_procesar_secuencial(rutas, tamano=IMAGE_SIZE):
    """Carga/preprocesa imagenes una por una en un solo proceso (baseline)."""
    imagenes = [_cargar_una_imagen(r, tamano) for r in rutas]
    return np.stack(imagenes)


def cargar_y_procesar_paralelo(rutas, tamano=IMAGE_SIZE, n_workers=None):
    """
    Misma carga/preprocesamiento, mas repartida entre los nucleos de CPU.
    Por defecto usa como mucho 8 procesos: en Windows cada worker (spawn)
    reimporta el modulo "__main__" que lo lanzo, y con maquinas de muchos
    nucleos (p.ej. 32 hilos) lanzar un Pool del tamano del CPU satura la
    memoria/pagefile del sistema antes de que alcance a ayudar en velocidad.
    """
    if n_workers is None:
        n_workers = min(4, os.cpu_count() or 1)
    with Pool(processes=n_workers) as pool:
        imagenes = pool.starmap(_cargar_una_imagen, [(r, tamano) for r in rutas])
    return np.stack(imagenes)


def preparar_dataset(train_dir, modo="paralelo", n_workers=None,
                      muestras_por_clase=MUESTRAS_POR_CLASE):
    """
    Reproduce el pipeline de datos del Laboratorio 3: muestreo de N
    imagenes/clase (600 por defecto, igual que el laboratorio original),
    split 70/15/15 estratificado, carga + resize a 64x64 + normalizacion
    [0,1]. Devuelve los arreglos listos para entrenar y el tiempo
    (segundos) que tomo la fase de carga/preprocesamiento.
    """
    clases, datos = listar_rutas_y_etiquetas(train_dir, muestras_por_clase=muestras_por_clase)
    train_df, val_df, test_df = dividir_train_val_test(datos)

    cargar = cargar_y_procesar_paralelo if modo == "paralelo" else cargar_y_procesar_secuencial

    inicio = time.perf_counter()
    kwargs = {"n_workers": n_workers} if modo == "paralelo" else {}
    X_train = cargar(train_df["ruta"].tolist(), **kwargs)
    X_val = cargar(val_df["ruta"].tolist(), **kwargs)
    X_test = cargar(test_df["ruta"].tolist(), **kwargs)
    tiempo_prep = time.perf_counter() - inicio

    etiqueta_a_indice = {letra: i for i, letra in enumerate(clases)}
    y_train = train_df["etiqueta"].map(etiqueta_a_indice).values
    y_val = val_df["etiqueta"].map(etiqueta_a_indice).values
    y_test = test_df["etiqueta"].map(etiqueta_a_indice).values

    return {
        "clases": clases,
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "tiempo_prep_segundos": tiempo_prep,
        "modo": modo,
    }


if __name__ == "__main__":
    import sys
    train_dir = sys.argv[1] if len(sys.argv) > 1 else "data/archive/asl_alphabet_train/asl_alphabet_train/"
    resultado = preparar_dataset(train_dir, modo="paralelo")
    print(f"X_train: {resultado['X_train'].shape}, tiempo de preparacion: {resultado['tiempo_prep_segundos']:.2f} s")
