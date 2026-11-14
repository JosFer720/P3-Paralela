"""
Reimplementacion en PyTorch de la CNN 2 del Laboratorio 3 (la arquitectura
que mejor resultado dio: 3 bloques convolucionales + dropout). Se usa
PyTorch en vez de TensorFlow/Keras porque TensorFlow dejo de dar soporte
nativo de GPU/CUDA en Windows a partir de la version 2.11 (requeriria WSL2),
mientras que PyTorch si corre con CUDA nativamente en Windows. La
arquitectura es equivalente para que la comparacion CPU vs GPU sea justa.
"""

import torch
import torch.nn as nn


class CNN2(nn.Module):
    def __init__(self, num_clases, image_size=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        tamano_reducido = image_size // 8  # 3 maxpool de /2
        self.clasificador = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * tamano_reducido * tamano_reducido, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_clases),
        )
        self._inicializar_pesos()

    def _inicializar_pesos(self):
        for modulo in self.modules():
            if isinstance(modulo, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(modulo.weight, nonlinearity="relu")
                nn.init.zeros_(modulo.bias)

    def forward(self, x):
        x = self.features(x)
        return self.clasificador(x)
