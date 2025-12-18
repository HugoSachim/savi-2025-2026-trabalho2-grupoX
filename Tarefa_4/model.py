from torchinfo import summary
import torch.nn as nn
import torch
import torch.nn.functional as F

class ModelDetectionOptimized(nn.Module):
    def __init__(self, max_digits=5):
        super(ModelDetectionOptimized, self).__init__()

        self.max_digits = max_digits

        # --- Extrator de Características (CNN) ---
        # Esta parte aprende a "ver" as formas dos números
        self.features = nn.Sequential(
            # Camada 1: 128x128 -> 64x64
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Camada 2: 64x64 -> 32x32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Camada 3: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Camada 4: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # O output da última camada convolucional será: 128 canais * 8x8 pixels
        n_flatten = 128 * 8 * 8 

        # --- Cabeça de Regressão (Localização) ---
        # Esta parte transforma as formas em coordenadas x, y, w, h
        self.locator = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Dropout(0.2), # Ajuda a não decorar o dataset (overfitting)
            nn.Linear(512, max_digits * 4),
            nn.Sigmoid() # Garante que as coordenadas saiam entre 0 e 1
        )

        print(f'Model Optimized initialized. Outputting {max_digits} boxes.')
        summary(self, input_size=(1, 1, 128, 128))

    def forward(self, x):
        # 1. Passa pela CNN (extração de formas)
        x = self.features(x)
        
        # 2. Achata para entrar na camada linear
        x = x.view(x.size(0), -1)
        
        # 3. Prediz as coordenadas
        y = self.locator(x)
        
        # 4. Organiza em [batch, max_digits, 4]
        # As coordenadas saem entre 0 e 1 por causa do Sigmoid
        y = y.view(-1, self.max_digits, 4)
        
        return y

    def getNumberOfParameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# model = ModelDetectionOptimized(max_digits=5)