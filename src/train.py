"""
Módulo de Entrenamiento para la U-Net de Segmentación de Núcleos.

Este script maneja el ciclo completo de entrenamiento, incluyendo:
- Carga de datos y dataloaders.
- Definición del modelo, función de pérdida y optimizador.
- Bucle de entrenamiento y validación.
- Guardado de checkpoints del modelo.
- Logging de métricas.

Autor: Proyecto de Segmentación de Núcleos
Fecha: 2025
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .conv_blocks import BasicUNet
from .nuclei_dataset import NucleiDataset

# --- Hiperparámetros y Configuración ---
HIPERPARAMETROS = {
    'learning_rate': 1e-4,
    'batch_size': 4,
    'num_epochs': 2, # Solo 2 epochs para generar modelo guardado
    'val_percent': 0.2, # 20% de datos para validación
    'data_path': '/home/tomy/Documents/ML-AI-Projects/data',
    'img_size': (256, 256)
}

def main():
    """Función principal para ejecutar el entrenamiento."""
    print("🚀 Iniciando el proceso de entrenamiento...")

    # 1. Configuración del dispositivo (GPU si está disponible)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo de entrenamiento: {device}")

    # 2. Carga y división del dataset
    try:
        dataset = NucleiDataset(data_root=HIPERPARAMETROS['data_path'], image_size=HIPERPARAMETROS['img_size'])
        n_val = int(len(dataset) * HIPERPARAMETROS['val_percent'])
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val],
                                            generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train_set, batch_size=HIPERPARAMETROS['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=HIPERPARAMETROS['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
        
        print(f"Dataset cargado: {len(dataset)} imágenes")
        print(f"   - Entrenamiento: {len(train_set)} imágenes en {len(train_loader)} lotes")
        print(f"   - Validación:    {len(val_set)} imágenes en {len(val_loader)} lotes")

    except (FileNotFoundError, ValueError) as e:
        print(f"Error al cargar el dataset: {e}")
        return

    # 3. Inicialización del modelo, función de pérdida y optimizador
    model = BasicUNet(in_channels=3, out_channels=1)
    model.to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=HIPERPARAMETROS['learning_rate'])
    criterion = nn.BCEWithLogitsLoss() # Ideal para segmentación binaria

    print("Modelo, optimizador y función de pérdida inicializados.")
    print(f"   - Modelo: BasicUNet con {sum(p.numel() for p in model.parameters()):,} parámetros")
    print(f"   - Optimizador: Adam, LR={HIPERPARAMETROS['learning_rate']}")
    print(f"   - Función de pérdida: BCEWithLogitsLoss")

    # 4. Bucle de entrenamiento
    for epoch in range(HIPERPARAMETROS['num_epochs']):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            images = batch['image'].to(device=device, dtype=torch.float32)
            true_masks = batch['mask'].unsqueeze(1).to(device=device, dtype=torch.float32)

            # Forward pass
            masks_pred = model(images)
            loss = criterion(masks_pred, true_masks)

            # Backward pass y optimización
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        # Evaluación
        val_score = evaluate(model, val_loader, device)
        model.train() # Volver a modo entrenamiento

        print(f"--- Epoch {epoch + 1}/{HIPERPARAMETROS['num_epochs']} ---")
        print(f"   Loss de entrenamiento: {epoch_loss / len(train_loader):.4f}")
        print(f"   Dice Score (validación): {val_score:.4f}")

    print("✅ Entrenamiento completado.")
    # Guardar el modelo entrenado
    torch.save(model.state_dict(), 'unet_nuclei.pth')
    print("Modelo guardado en unet_nuclei.pth")


def dice_coeff(pred, target, smooth=1.0):
    """Calcula el coeficiente de Dice para un batch."""
    pred = torch.sigmoid(pred) # Aplicar sigmoide para obtener probabilidades
    pred = (pred > 0.5).float() # Binarizar la predicción
    
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()


def evaluate(model, dataloader, device):
    """Evalúa el modelo en el set de validación."""
    model.eval() # Poner el modelo en modo evaluación
    num_val_batches = len(dataloader)
    dice_score = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device=device, dtype=torch.float32)
            masks = batch['mask'].unsqueeze(1).to(device=device, dtype=torch.float32)
            
            pred_masks = model(images)
            dice_score += dice_coeff(pred_masks, masks)

    model.train() # Reactivar modo entrenamiento
    return dice_score / num_val_batches


if __name__ == '__main__':
    main()
