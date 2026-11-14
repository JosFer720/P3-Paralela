# Proyecto 3: Mejorando algoritmos con CUDA

**CC3069 — Computación Paralela y Distribuida, UVG, Semestre II 2026**

Integrantes: Fernando Ruíz, Olivier Viau, Jose López.

## Contexto

Este proyecto toma la solución secuencial desarrollada en el **Laboratorio 3 de
Data Science (CC3084)** — un clasificador de letras del alfabeto ASL a partir
de imágenes de manos, usando el dataset [ASL Alphabet de
Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) — y la
reimplementa separando explícitamente:

- **Preparación de imágenes → CPU.** Lectura de archivos, decodificación JPEG,
  redimensionamiento y normalización son operaciones de I/O independientes
  por imagen, no aritmética masiva: se paralelizan entre los núcleos de la
  CPU (`multiprocessing`), no en GPU.
- **Entrenamiento del modelo → GPU con CUDA.** Las convoluciones, el
  producto de matrices y el backpropagation sí son aritmética masivamente
  paralela: se delegan al GPU.

El código original del Laboratorio 3 (Hugo Barillas, Jose López, Luis
Palacios) está en
[Hugo2049/Laboratorio-3-data-science](https://github.com/Hugo2049/Laboratorio-3-data-science)
y sirve como baseline secuencial y como referencia de la arquitectura
reutilizada (CNN 2: la de mejor resultado, 96.3% de exactitud en validación).

**Nota técnica:** el Laboratorio 3 usó TensorFlow/Keras, pero TensorFlow dejó
de dar soporte nativo de GPU en Windows desde la versión 2.11 (requeriría
correr todo dentro de WSL2). Por eso este proyecto reimplementa la misma
arquitectura de CNN 2 en **PyTorch**, que sí corre con CUDA nativamente en
Windows, para que la comparación CPU vs. GPU sea directa sin depender de
WSL2. La arquitectura, los hiperparámetros y el split de datos son los
mismos que en el laboratorio original.

## Paso 1: análisis de la solución secuencial

### Partes principales del programa (Laboratorio3_ASL.ipynb)

| # | Parte | Función |
|---|-------|---------|
| 1 | Carga de rutas y muestreo | Lista las 29 clases, toma 600 imágenes/clase (17,400 en total) y arma el DataFrame de rutas + etiquetas |
| 2 | Split train/val/test | `train_test_split` estratificado 70/15/15 sobre las rutas |
| 3 | Carga y preprocesamiento | Lee cada imagen de disco, la redimensiona 200x200 → 64x64 y normaliza a [0,1] |
| 4 | Definición de modelos | Arquitecturas Keras: CNN1, CNN2, red fully-connected |
| 5 | Entrenamiento | `model.fit` — forward pass, cálculo de pérdida, backward pass, actualización de pesos, por batch y por época |
| 6 | Evaluación | `model.evaluate` / `predict` sobre validación, prueba y fotos propias |

### Operaciones repetitivas

| Operación repetitiva | Datos que procesa | Cantidad de repeticiones |
|---|---|---|
| Lectura + resize + normalización de una imagen | 1 imagen JPEG 200x200x3 → arreglo 64x64x3 | 17,400 (600 imágenes × 29 clases) |
| Forward pass de un batch | Tensor de batch (batch_size, 64, 64, 3) a través de 3 bloques conv + densas | (12,180 / batch_size) × épocas |
| Backward pass de un batch | Gradientes de la pérdida respecto a todos los pesos del modelo | (12,180 / batch_size) × épocas |
| Convolución 2D dentro de un bloque | Cada filtro sobre cada posición espacial del mapa de activación | filtros × alto × ancho × canales de entrada, por batch, por época |

Con `batch_size=32` y `epocas=10` sobre 12,180 imágenes de entrenamiento, el
forward+backward se repite ≈ 381 veces por época, ≈ 3,810 veces en total —
y cada una de esas pasadas internamente dispara millones de multiplicaciones
en las convoluciones. Esa es la operación que de verdad justifica el GPU.

## Estructura del repositorio

```
src/
  data_prep.py    # Fase CPU: muestreo, split, carga/resize/normalización (secuencial y multi-núcleo)
  model.py        # CNN2 reimplementada en PyTorch (misma arquitectura que el Lab 3)
  train.py        # Entrenamiento parametrizado por device ("cpu" / "cuda")
  benchmark.py     # Corre todo el pipeline y calcula los speedups
notebooks/
  Proyecto3_CUDA.ipynb  # Informe: análisis, resultados, gráficos, discusión
data/
  archive/        # Acá va el dataset descargado de Kaggle (ignorado por git, pesa >1GB)
results/
  benchmark_resultados.json  # Tiempos y speedups de la última corrida
```

## Cómo correrlo

1. Descargar el dataset [ASL Alphabet de
   Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) y
   descomprimirlo en `data/archive/` (debe quedar
   `data/archive/asl_alphabet_train/asl_alphabet_train/<letra>/*.jpg`).
2. Instalar dependencias: `pip install -r requirements.txt`. Para usar GPU,
   instalar la build de PyTorch con CUDA que corresponda a tu driver (ver
   [pytorch.org/get-started](https://pytorch.org/get-started/locally/)).
3. Correr el benchmark completo:

   ```bash
   python -m src.benchmark --train_dir data/archive/asl_alphabet_train/asl_alphabet_train/
   ```

   Esto mide: preparación de datos (CPU secuencial vs. multi-núcleo) y
   entrenamiento de CNN2 (CPU vs. GPU si hay una disponible), y guarda los
   resultados en `results/benchmark_resultados.json`.
4. Durante la corrida, capturar pantallazos de `nvidia-smi` (o el
   Administrador de Tareas) mostrando el uso de CPU en la fase de
   preparación y el uso de GPU en la fase de entrenamiento, para el informe.
5. Documentar los resultados, speedups y discusión en
   `notebooks/Proyecto3_CUDA.ipynb`.
