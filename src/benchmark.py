"""
Orquesta el benchmark completo del Proyecto 3:
  1. Preparacion de imagenes en CPU (secuencial vs multi-nucleo).
  2. Entrenamiento de CNN2 en CPU vs GPU (CUDA).
  3. Calculo de speedup por fase y guardado de resultados en results/.

Uso:
    python -m src.benchmark --train_dir data/archive/asl_alphabet_train/asl_alphabet_train/
"""

import argparse
import json
import os

# NOTA: torch y src.train se importan de forma perezosa (dentro de las
# funciones), despues de que termina la fase de multiprocessing. En Windows,
# multiprocessing.Pool (spawn) reconstruye este modulo "__main__" dentro de
# cada proceso hijo, lo que ejecutaria "import torch" (con sus DLLs de CUDA)
# una vez por cada worker. Con muchos nucleos eso agota la memoria/pagefile
# del sistema y los workers truenan al cargar las DLLs. Importando torch solo
# despues de que el Pool ya termino, los workers nunca necesitan cargarlo.
from src.data_prep import preparar_dataset


def correr_benchmark(train_dir, epocas=10, batch_size=32, salida_dir="results", n_workers=None):
    os.makedirs(salida_dir, exist_ok=True)
    resultados = {}

    # --- Fase 1: preparacion de datos (CPU secuencial vs CPU multi-nucleo) ---
    print("== Preparando datos (CPU, secuencial) ==")
    datos_seq = preparar_dataset(train_dir, modo="secuencial")
    print("== Preparando datos (CPU, multi-nucleo) ==")
    datos_par = preparar_dataset(train_dir, modo="paralelo", n_workers=n_workers)

    import torch  # import perezoso: recien aqui ya no hay mas Pool() por venir
    from src.train import entrenar

    resultados["prep_cpu_secuencial_s"] = datos_seq["tiempo_prep_segundos"]
    resultados["prep_cpu_multinucleo_s"] = datos_par["tiempo_prep_segundos"]
    resultados["speedup_prep"] = (
        datos_seq["tiempo_prep_segundos"] / datos_par["tiempo_prep_segundos"]
    )

    clases = datos_par["clases"]
    X_train, y_train = datos_par["X_train"], datos_par["y_train"]
    X_val, y_val = datos_par["X_val"], datos_par["y_val"]

    # --- Fase 2: entrenamiento (CPU vs GPU/CUDA) ---
    print("== Entrenando CNN2 en CPU ==")
    _, _, tiempo_cpu = entrenar(
        X_train, y_train, X_val, y_val, len(clases),
        device="cpu", epocas=epocas, batch_size=batch_size,
    )
    resultados["entrenamiento_cpu_s"] = tiempo_cpu

    if torch.cuda.is_available():
        print("== Entrenando CNN2 en GPU (CUDA) ==")
        _, _, tiempo_gpu = entrenar(
            X_train, y_train, X_val, y_val, len(clases),
            device="cuda", epocas=epocas, batch_size=batch_size,
        )
        resultados["entrenamiento_gpu_s"] = tiempo_gpu
        resultados["speedup_entrenamiento"] = tiempo_cpu / tiempo_gpu
        resultados["gpu_nombre"] = torch.cuda.get_device_name(0)
    else:
        print("No se detecto GPU con CUDA disponible; se omite la fase GPU.")

    ruta_salida = os.path.join(salida_dir, "benchmark_resultados.json")
    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    print("\n=== Resumen ===")
    for clave, valor in resultados.items():
        print(f"{clave}: {valor}")
    print(f"\nGuardado en {ruta_salida}")
    return resultados


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dir", default="data/archive/asl_alphabet_train/asl_alphabet_train/"
    )
    parser.add_argument("--epocas", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_workers", type=int, default=None,
                         help="Nucleos de CPU a usar en la fase de preparacion (default: ver data_prep)")
    args = parser.parse_args()

    correr_benchmark(args.train_dir, epocas=args.epocas, batch_size=args.batch_size, n_workers=args.n_workers)
