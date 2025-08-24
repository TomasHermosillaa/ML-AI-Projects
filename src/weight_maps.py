"""
Módulo para la generación de mapas de peso y detección de bordes para la segmentación
de núcleos celulares.

Este módulo implementa funciones para:
1. Detectar bordes usando operaciones morfológicas
2. Generar mapas de peso basados en distancias a bordes y balance de clases
3. Visualizar los mapas generados
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import morphology, segmentation, feature
import cv2

def detect_edges(mask, kernel_size=3):
    """
    Detecta los bordes de los núcleos utilizando operaciones morfológicas.
    
    Args:
        mask: Numpy array de la máscara binaria (0=fondo, 255=núcleo)
        kernel_size: Tamaño del elemento estructural para operaciones morfológicas
    
    Returns:
        edges: Mapa binario de los bordes (1=borde, 0=no borde)
    """
    # Normalizar máscara a valores binarios 0 y 1
    mask_binary = mask.copy()
    if mask_binary.max() > 1:
        mask_binary = mask_binary / 255
    
    # Crear elemento estructural para operaciones morfológicas
    selem = morphology.disk(kernel_size // 2)
    
    # Aplicar erosión para reducir tamaño de los núcleos
    mask_eroded = morphology.erosion(mask_binary, selem)
    
    # Los bordes son la diferencia entre la máscara original y la erosionada
    edges = mask_binary - mask_eroded
    
    return edges

def detect_inner_edges(mask, kernel_size=3):
    """
    Detecta específicamente los bordes internos entre núcleos adyacentes.
    
    Args:
        mask: Numpy array de la máscara binaria (0=fondo, 255=núcleo)
        kernel_size: Tamaño del elemento estructural para operaciones morfológicas
    
    Returns:
        inner_edges: Mapa binario de los bordes internos (1=borde interno, 0=no borde)
    """
    # Normalizar máscara a valores binarios 0 y 1
    mask_binary = mask.copy()
    if mask_binary.max() > 1:
        mask_binary = mask_binary / 255
    
    # Detectar todos los bordes
    all_edges = detect_edges(mask_binary, kernel_size)
    
    # Aplicar dilatación para expandir núcleos
    selem = morphology.disk(kernel_size // 2)
    mask_dilated = morphology.dilation(mask_binary, selem)
    
    # Bordes externos (perímetro exterior de los núcleos)
    outer_edges = mask_dilated - mask_binary
    
    # Los bordes internos son los que no son bordes externos y están en todos los bordes
    inner_edges = all_edges * (1 - outer_edges)
    
    return inner_edges

def generate_weight_map(mask, w0=10, sigma=5):
    """
    Genera un mapa de pesos basado en la distancia a los bordes y balance de clases.
    
    Args:
        mask: Numpy array de la máscara binaria (0=fondo, 255=núcleo)
        w0: Peso base para los bordes
        sigma: Parámetro de decaimiento de la función de peso
    
    Returns:
        weight_map: Mapa de pesos para entrenamiento
    """
    # Normalizar máscara a valores binarios 0 y 1
    mask_binary = mask.copy()
    if mask_binary.max() > 1:
        mask_binary = mask_binary / 255
    
    # Detectar bordes, con enfoque en los bordes internos
    edges = detect_edges(mask_binary)
    inner_edges = detect_inner_edges(mask_binary)
    
    # Combinar bordes dando mayor peso a los internos
    combined_edges = edges + 2 * inner_edges
    combined_edges = np.clip(combined_edges, 0, 1)
    
    # Calcular mapa de distancia a los bordes
    if combined_edges.sum() > 0:  # Verificar que existan bordes
        dist_transform = ndimage.distance_transform_edt(1 - combined_edges)
        
        # Función de peso basada en distancia
        weight_map = w0 * np.exp(-(dist_transform**2) / (2 * (sigma**2)))
    else:
        # Si no hay bordes, usar un mapa plano
        weight_map = np.ones_like(mask_binary)
    
    # Añadir factor de balance de clases
    # Calculamos el ratio de fondo vs núcleos
    foreground_ratio = np.mean(mask_binary)
    background_ratio = 1 - foreground_ratio
    
    # Aplicar pesos por clase para contrarrestar el desbalance
    class_weight = np.ones_like(mask_binary)
    if foreground_ratio < background_ratio:
        # Si hay menos píxeles de núcleo, les damos más peso
        class_balance_factor = background_ratio / foreground_ratio
        class_weight[mask_binary > 0] = class_balance_factor
    
    # Combinar mapa de pesos por distancia con balance de clases
    final_weight_map = weight_map * class_weight
    
    # Normalizar para mantener un rango manejable
    final_weight_map = final_weight_map / final_weight_map.mean()
    
    return final_weight_map

def visualize_weight_map(image, mask, weight_map, show_edges=True):
    """
    Visualiza la imagen original, la máscara y el mapa de pesos.
    
    Args:
        image: Imagen original RGB
        mask: Máscara binaria de segmentación
        weight_map: Mapa de pesos generado
        show_edges: Si se deben visualizar los bordes detectados
    """
    # Normalizar máscara a valores binarios 0 y 1
    mask_binary = mask.copy()
    if mask_binary.max() > 1:
        mask_binary = mask_binary / 255
        
    # Crear figura con subplots
    fig, axes = plt.subplots(1, 4 if show_edges else 3, figsize=(18, 5))
    
    # Imagen original
    axes[0].imshow(image)
    axes[0].set_title('Imagen Original')
    axes[0].axis('off')
    
    # Máscara de segmentación
    axes[1].imshow(mask_binary, cmap='gray')
    axes[1].set_title('Máscara de Segmentación')
    axes[1].axis('off')
    
    # Mapa de pesos
    im = axes[2].imshow(weight_map, cmap='viridis')
    axes[2].set_title('Mapa de Pesos')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    if show_edges:
        # Bordes detectados
        edges = detect_edges(mask_binary)
        inner_edges = detect_inner_edges(mask_binary)
        
        # Visualizar bordes sobre la imagen original
        overlay = image.copy()
        if len(overlay.shape) == 2:  # Si es imagen en escala de grises
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
        
        # Normalizar a [0, 1] si es necesario
        if overlay.max() > 1:
            overlay = overlay / 255.0
            
        # Pintar bordes en la imagen
        overlay[edges == 1, 0] = 1.0  # Rojo para todos los bordes
        overlay[edges == 1, 1:] = 0.0
        
        overlay[inner_edges == 1, 0] = 1.0  # Amarillo para bordes internos
        overlay[inner_edges == 1, 1] = 1.0
        overlay[inner_edges == 1, 2] = 0.0
        
        axes[3].imshow(overlay)
        axes[3].set_title('Bordes Detectados')
        axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return fig
