"""
Entrenamiento de CNN2 parametrizado por dispositivo (cpu/cuda). Esta es la
fase "secuencial vs paralela" del proyecto: la matematica del entrenamiento
(convoluciones, multiplicaciones de matrices, backprop) es la que se delega
al GPU via CUDA (a traves de cuBLAS/cuDNN, que PyTorch invoca internamente
cuando el modelo y los tensores viven en device="cuda"). La preparacion de
imagenes (src/data_prep.py) nunca pasa por aqui: se hace antes, en CPU.
"""

import time

import numpy as np
import torch
import torch.nn as nn

from src.model import CNN2


def _a_tensores(X, y, device):
    # (N, H, W, C) -> (N, C, H, W), que es el formato que espera Conv2d de PyTorch
    X_t = torch.from_numpy(X).permute(0, 3, 1, 2).float().to(device)
    y_t = torch.from_numpy(np.asarray(y)).long().to(device)
    return X_t, y_t


def entrenar(X_train, y_train, X_val, y_val, num_clases, device="cpu",
             epocas=10, batch_size=32, lr=1e-3, seed=42):
    """
    Entrena CNN2 en el device indicado y devuelve el modelo entrenado, el
    historial de metricas por epoca y el tiempo total de entrenamiento.
    Usar device="cpu" reproduce el baseline secuencial; device="cuda"
    corre exactamente el mismo codigo pero delega la matematica a la GPU.
    """
    torch.manual_seed(seed)
    device = torch.device(device)

    modelo = CNN2(num_clases).to(device)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    criterio = nn.CrossEntropyLoss()

    X_train_t, y_train_t = _a_tensores(X_train, y_train, device)
    X_val_t, y_val_t = _a_tensores(X_val, y_val, device)

    n = X_train_t.shape[0]
    historial = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    inicio = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize()

    for epoca in range(epocas):
        modelo.train()
        permutacion = torch.randperm(n, device=device)
        perdida_epoca, correctos_epoca = 0.0, 0

        for inicio_lote in range(0, n, batch_size):
            idx = permutacion[inicio_lote:inicio_lote + batch_size]
            X_lote, y_lote = X_train_t[idx], y_train_t[idx]

            optimizador.zero_grad()
            salida = modelo(X_lote)
            perdida = criterio(salida, y_lote)
            perdida.backward()
            optimizador.step()

            perdida_epoca += perdida.item() * X_lote.size(0)
            correctos_epoca += (salida.argmax(1) == y_lote).sum().item()

        modelo.eval()
        with torch.no_grad():
            salida_val = modelo(X_val_t)
            perdida_val = criterio(salida_val, y_val_t).item()
            acc_val = (salida_val.argmax(1) == y_val_t).float().mean().item()

        historial["train_loss"].append(perdida_epoca / n)
        historial["train_acc"].append(correctos_epoca / n)
        historial["val_loss"].append(perdida_val)
        historial["val_acc"].append(acc_val)

        print(f"[{device}] epoca {epoca + 1}/{epocas} - "
              f"train_acc={correctos_epoca / n:.4f} - val_acc={acc_val:.4f}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    tiempo_entrenamiento = time.perf_counter() - inicio

    return modelo, historial, tiempo_entrenamiento
