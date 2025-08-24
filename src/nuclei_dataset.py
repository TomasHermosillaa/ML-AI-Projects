"""
Dataset personalizado para segmentación de núcleos celulares.

Este módulo implementa un Dataset de PyTorch que integra:
- Carga de imágenes y máscaras desde el dataset DSB2018
- Generación de mapas de peso usando operaciones morfológicas
- Aplicación de técnicas de normalización
- Transformaciones y conversión a tensores PyTorch

Autor: Proyecto de Segmentación de Núcleos
Fecha: 2025
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
from typing import Tuple, Optional, Callable
import logging

# Importar funciones desarrolladas en actividades anteriores
from .normalization import (
    normalize_per_channel,
    normalize_clahe, 
    normalize_gamma_correction
)


class NucleiDataset(Dataset):
    """
    Dataset personalizado para imágenes de núcleos celulares.
    
    Este dataset maneja:
    - Carga automática de imágenes RGB y máscaras binarias
    - Generación de mapas de peso para enfatizar bordes
    - Aplicación de normalizaciones configurables
    - Transformaciones para aumento de datos (opcional)
    """
    
    def __init__(self, 
                 data_root: str,
                 split: str = 'train',
                 image_size: Tuple[int, int] = (256, 256),
                 normalization_method: str = 'per_channel',
                 generate_weight_maps: bool = True,
                 weight_params: Optional[dict] = None,
                 transform: Optional[Callable] = None,
                 seed: int = 42):
        """
        Inicializa el dataset de núcleos celulares.
        
        Args:
            data_root: Directorio raíz que contiene las carpetas de muestras
            split: Tipo de split ('train', 'val', 'test') - por implementar
            image_size: Tamaño objetivo para redimensionar imágenes
            normalization_method: Método de normalización ('per_channel', 'clahe_gamma', 'none')
            generate_weight_maps: Si generar mapas de peso morfológicos
            weight_params: Parámetros para generación de mapas de peso
            transform: Transformaciones adicionales (aumento de datos)
            seed: Semilla para reproducibilidad
        """
        super().__init__()
        
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.normalization_method = normalization_method
        self.generate_weight_maps = generate_weight_maps
        self.transform = transform
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Parámetros por defecto para mapas de peso
        default_weight_params = {
            'background_weight': 0.1,
            'foreground_weight': 1.0, 
            'edge_weight': 5.0,
            'edge_kernel_size': 5
        }
        self.weight_params = weight_params or default_weight_params
        
        # Establecer semilla para reproducibilidad
        random.seed(seed)
        np.random.seed(seed)
        
        # Buscar y cargar muestras disponibles
        self._load_sample_paths()
        
        self.logger.info(f"Inicializado NucleiDataset con {len(self.sample_paths)} muestras")
        self.logger.info(f"Configuración: normalización='{normalization_method}', "
                        f"mapas_peso={generate_weight_maps}, tamaño={image_size}")
    
    def _load_sample_paths(self):
        """
        Busca y carga los paths de todas las muestras disponibles.
        """
        self.sample_paths = []
        
        if not self.data_root.exists():
            raise FileNotFoundError(f"Directorio de datos no encontrado: {self.data_root}")
        
        # Buscar carpetas con estructura: data/00001/, data/00002/, etc.
        for sample_dir in self.data_root.iterdir():
            if sample_dir.is_dir():
                image_path = sample_dir / 'image.png'
                mask_path = sample_dir / 'mask.png'
                
                # Verificar que existen ambos archivos
                if image_path.exists() and mask_path.exists():
                    self.sample_paths.append({
                        'sample_id': sample_dir.name,
                        'image_path': image_path,
                        'mask_path': mask_path
                    })
                else:
                    self.logger.warning(f"Muestra incompleta ignorada: {sample_dir.name}")
        
        if len(self.sample_paths) == 0:
            raise ValueError(f"No se encontraron muestras válidas en {self.data_root}")
        
        # Ordenar para consistencia
        self.sample_paths.sort(key=lambda x: x['sample_id'])
    
    def __len__(self) -> int:
        """
        Retorna el número total de muestras en el dataset.
        """
        return len(self.sample_paths)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Retorna una muestra del dataset.
        
        Args:
            idx: Índice de la muestra a obtener
            
        Returns:
            dict con claves:
                - 'image': Tensor de imagen (C, H, W)
                - 'mask': Tensor de máscara (H, W)
                - 'weight_map': Tensor de pesos (H, W) si generate_weight_maps=True
                - 'sample_id': ID de la muestra (string)
        """
        if idx >= len(self.sample_paths):
            raise IndexError(f"Índice {idx} fuera de rango (dataset tiene {len(self)} muestras)")
        
        sample_info = self.sample_paths[idx]
        
        try:
            # 1. Cargar imagen y máscara
            image = self._load_image(sample_info['image_path'])
            mask = self._load_mask(sample_info['mask_path'])
            
            # 2. Redimensionar si es necesario
            if image.shape[:2] != self.image_size:
                image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
            
            # 3. Aplicar normalización
            image = self._apply_normalization(image)
            
            # 4. Generar mapa de pesos si está habilitado
            weight_map = None
            if self.generate_weight_maps:
                weight_map = self._generate_weight_map(mask)
            
            # 5. Aplicar transformaciones adicionales si existen
            if self.transform is not None:
                # Las transformaciones deben manejar image, mask y weight_map juntos
                transformed = self.transform(image=image, mask=mask, weight_map=weight_map)
                image = transformed['image']
                mask = transformed['mask']
                if weight_map is not None:
                    weight_map = transformed.get('weight_map', weight_map)
            
            # 6. Convertir a tensores PyTorch
            image_tensor = self._to_tensor_image(image)
            mask_tensor = self._to_tensor_mask(mask)
            
            # Preparar salida
            sample = {
                'image': image_tensor,
                'mask': mask_tensor,
                'sample_id': sample_info['sample_id']
            }
            
            if weight_map is not None:
                weight_tensor = torch.from_numpy(weight_map).float()
                sample['weight_map'] = weight_tensor
            
            return sample
            
        except Exception as e:
            self.logger.error(f"Error cargando muestra {sample_info['sample_id']}: {str(e)}")
            raise
    
    def _load_image(self, image_path: Path) -> np.ndarray:
        """
        Carga una imagen RGB desde archivo.
        
        Returns:
            numpy array con shape (H, W, 3) y valores [0, 255]
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        # Convertir BGR a RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def _load_mask(self, mask_path: Path) -> np.ndarray:
        """
        Carga una máscara binaria desde archivo.
        
        Returns:
            numpy array con shape (H, W) y valores [0, 255]
        """
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"No se pudo cargar la máscara: {mask_path}")
        
        return mask
    
    def _apply_normalization(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica el método de normalización configurado.
        
        Args:
            image: Imagen RGB (H, W, 3) con valores [0, 255]
            
        Returns:
            Imagen normalizada
        """
        if self.normalization_method == 'per_channel':
            return normalize_per_channel(image)
        
        elif self.normalization_method == 'clahe_gamma':
            # Combinar CLAHE + corrección gamma (de análisis previo)
            clahe_img = normalize_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8))
            return normalize_gamma_correction(clahe_img, gamma=0.7)
        
        elif self.normalization_method == 'none':
            # Solo convertir a float [0, 1]
            return image.astype(np.float32) / 255.0
        
        else:
            raise ValueError(f"Método de normalización no reconocido: {self.normalization_method}")
    
    def _generate_weight_map(self, mask: np.ndarray) -> np.ndarray:
        """
        Genera mapa de pesos usando operaciones morfológicas.
        
        Args:
            mask: Máscara binaria (H, W) con valores [0, 255]
            
        Returns:
            Mapa de pesos (H, W) como numpy array float32
        """
        # Binarizar máscara
        binary_mask = (mask > 127).astype(np.uint8)
        
        # Crear elemento estructurante para detectar bordes
        kernel_size = self.weight_params['edge_kernel_size']
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Calcular gradiente morfológico para detectar bordes
        gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
        edges_binary = (gradient > 0).astype(np.uint8)
        
        # Inicializar mapa de pesos
        weight_map = np.full(mask.shape, self.weight_params['background_weight'], dtype=np.float32)
        
        # Asignar pesos por región
        weight_map[binary_mask == 1] = self.weight_params['foreground_weight']  # Núcleos
        weight_map[edges_binary == 1] = self.weight_params['edge_weight']       # Bordes
        
        return weight_map
    
    def _to_tensor_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Convierte imagen numpy a tensor PyTorch.
        
        Args:
            image: Imagen (H, W, 3) normalizada
            
        Returns:
            Tensor (3, H, W)
        """
        # Transponer de (H, W, C) a (C, H, W)
        if len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))
        
        return torch.from_numpy(image).float()
    
    def _to_tensor_mask(self, mask: np.ndarray) -> torch.Tensor:
        """
        Convierte máscara numpy a tensor PyTorch.
        
        Args:
            mask: Máscara (H, W) con valores [0, 255]
            
        Returns:
            Tensor (H, W) con valores [0, 1]
        """
        # Binarizar y convertir a float
        binary_mask = (mask > 127).astype(np.float32)
        return torch.from_numpy(binary_mask)


def create_data_loaders(data_root: str,
                       batch_size: int = 4,
                       num_workers: int = 2,
                       train_split: float = 0.8,
                       val_split: float = 0.2,
                       **dataset_kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    Crea DataLoaders de entrenamiento y validación.
    
    Args:
        data_root: Directorio raíz de datos
        batch_size: Tamaño de batch
        num_workers: Número de workers para carga paralela
        train_split: Proporción de datos para entrenamiento
        val_split: Proporción de datos para validación
        **dataset_kwargs: Argumentos adicionales para NucleiDataset
        
    Returns:
        Tupla (train_loader, val_loader)
    """
    # Por ahora, usar todo el dataset como entrenamiento
    # En futuras iteraciones implementaremos splits apropiados
    
    train_dataset = NucleiDataset(
        data_root=data_root,
        split='train',
        **dataset_kwargs
    )
    
    # Crear una pequeña validación tomando una muestra del training
    val_size = max(1, int(len(train_dataset) * 0.1))  # 10% para validación
    train_size = len(train_dataset) - val_size
    
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Script de prueba para validar el dataset
    import matplotlib.pyplot as plt
    
    # Configurar paths
    data_root = "../data"  # Ajustar según ubicación
    
    # Crear dataset
    dataset = NucleiDataset(
        data_root=data_root,
        normalization_method='per_channel',
        generate_weight_maps=True
    )
    
    print(f"Dataset cargado con {len(dataset)} muestras")
    
    # Probar carga de una muestra
    sample = dataset[0]
    print(f"Muestra 0:")
    print(f"  - Image shape: {sample['image'].shape}")
    print(f"  - Mask shape: {sample['mask'].shape}")
    if 'weight_map' in sample:
        print(f"  - Weight map shape: {sample['weight_map'].shape}")
    print(f"  - Sample ID: {sample['sample_id']}")
    
    # Crear DataLoader de prueba
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    # Probar carga de batch
    batch = next(iter(loader))
    print(f"\nBatch de prueba:")
    print(f"  - Imágenes: {batch['image'].shape}")
    print(f"  - Máscaras: {batch['mask'].shape}")
    if 'weight_map' in batch:
        print(f"  - Mapas peso: {batch['weight_map'].shape}")
