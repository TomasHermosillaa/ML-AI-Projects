"""
Módulo de normalización de imágenes para segmentación de núcleos celulares.

Este módulo contiene diversas técnicas de normalización para imágenes biomédicas, 
incluyendo normalización estadística, por canal, ecualización de histograma, 
ecualización adaptativa, corrección gamma y normalización de Macenko.
"""

import numpy as np
import cv2
from skimage import exposure, color, util
from scipy import linalg
import matplotlib.pyplot as plt

def normalize_statistical(image, epsilon=1e-8):
    """
    Normaliza la imagen para tener media 0 y desviación estándar 1.
    
    Args:
        image: Imagen de entrada (RGB o escala de grises)
        epsilon: Pequeño valor para evitar división por cero
        
    Returns:
        Imagen normalizada con valores flotantes
    """
    # Convertir a float para cálculos precisos
    image_float = image.astype(np.float32)
    
    # Calcular media y desviación estándar
    mean = np.mean(image_float)
    std = np.std(image_float) + epsilon  # Añadir epsilon para evitar división por cero
    
    # Normalizar: (x - μ) / σ
    normalized = (image_float - mean) / std
    
    return normalized

def normalize_per_channel(image, epsilon=1e-8):
    """
    Normaliza cada canal de la imagen por separado para tener media 0 y desviación estándar 1.
    
    Args:
        image: Imagen de entrada RGB
        epsilon: Pequeño valor para evitar división por cero
        
    Returns:
        Imagen normalizada con valores flotantes
    """
    # Verificar que la imagen es RGB
    if len(image.shape) < 3 or image.shape[2] != 3:
        raise ValueError("Esta función requiere una imagen RGB de 3 canales")
    
    # Convertir a float para cálculos precisos
    image_float = image.astype(np.float32)
    normalized = np.zeros_like(image_float)
    
    # Normalizar cada canal por separado
    for c in range(3):
        channel = image_float[:,:,c]
        mean = np.mean(channel)
        std = np.std(channel) + epsilon
        normalized[:,:,c] = (channel - mean) / std
    
    return normalized

def normalize_minmax(image, target_range=(0, 1)):
    """
    Normaliza la imagen escalando valores a un rango específico (por defecto [0,1]).
    
    Args:
        image: Imagen de entrada
        target_range: Tuple (min, max) para el rango objetivo
        
    Returns:
        Imagen normalizada en el rango objetivo
    """
    # Convertir a float para cálculos precisos
    image_float = image.astype(np.float32)
    
    # Obtener mínimo y máximo actuales
    min_val = np.min(image_float)
    max_val = np.max(image_float)
    
    # Evitar división por cero
    if max_val == min_val:
        return np.ones_like(image_float) * target_range[0]
    
    # Normalizar al rango [0, 1]
    normalized = (image_float - min_val) / (max_val - min_val)
    
    # Escalar al rango objetivo
    normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
    
    return normalized

def normalize_histogram_equalization(image):
    """
    Aplica ecualización de histograma a cada canal de la imagen.
    
    Args:
        image: Imagen de entrada RGB
        
    Returns:
        Imagen con histograma ecualizado (valores en [0,1])
    """
    # Verificar que la imagen es RGB
    if len(image.shape) < 3 or image.shape[2] != 3:
        # Para imágenes en escala de grises
        return exposure.equalize_hist(image)
    
    # Para imágenes RGB, equalizar cada canal por separado
    result = np.zeros_like(image, dtype=np.float32)
    
    for c in range(3):
        result[:,:,c] = exposure.equalize_hist(image[:,:,c])
    
    return result

def normalize_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Aplica ecualización de histograma adaptativa con limitación de contraste (CLAHE).
    
    Args:
        image: Imagen de entrada RGB o escala de grises
        clip_limit: Límite de recorte para CLAHE (default: 2.0)
        tile_grid_size: Tamaño de la cuadrícula para CLAHE (default: 8x8)
        
    Returns:
        Imagen con CLAHE aplicado (valores en [0,1] para float, [0,255] para uint8)
    """
    # Verificar si la imagen es RGB o escala de grises
    if len(image.shape) < 3 or image.shape[2] != 3:
        # Imagen en escala de grises
        return exposure.equalize_adapthist(
            image, 
            clip_limit=clip_limit, 
            nbins=256
        )
    
    # Para imágenes RGB
    # Convertir a espacio de color LAB (L: luminosidad, a/b: cromaticidad)
    lab = color.rgb2lab(image.astype(np.float32) / 255.0)
    
    # Aplicar CLAHE solo al canal de luminosidad (L)
    # Primero normalizamos L al rango [0,1] para que CLAHE funcione correctamente
    l_channel = lab[:,:,0]
    l_channel_norm = (l_channel - l_channel.min()) / (l_channel.max() - l_channel.min())
    
    # Aplicar CLAHE
    l_channel_eq = exposure.equalize_adapthist(
        l_channel_norm, 
        clip_limit=clip_limit, 
        nbins=256
    )
    
    # Volver a escalar al rango original de L en LAB
    l_channel_min, l_channel_max = l_channel.min(), l_channel.max()
    l_channel_eq = l_channel_eq * (l_channel_max - l_channel_min) + l_channel_min
    
    # Reemplazar el canal L
    lab_eq = lab.copy()
    lab_eq[:,:,0] = l_channel_eq
    
    # Convertir de vuelta a RGB
    result = color.lab2rgb(lab_eq)
    
    # Si la entrada era uint8, convertir el resultado a uint8
    if image.dtype == np.uint8:
        result = (result * 255).astype(np.uint8)
        
    return result

def normalize_gamma_correction(image, gamma=1.0):
    """
    Aplica corrección gamma a la imagen.
    
    Args:
        image: Imagen de entrada
        gamma: Valor gamma (default: 1.0 - sin cambios)
            - gamma < 1: Aclara la imagen
            - gamma > 1: Oscurece la imagen
            
    Returns:
        Imagen con corrección gamma aplicada
    """
    # Asegurar que la imagen está en el rango [0,1]
    if image.dtype == np.uint8:
        image_float = image.astype(np.float32) / 255.0
    else:
        # Si ya está en float, aseguramos que esté en [0,1]
        image_float = np.clip(image.astype(np.float32), 0, 1)
    
    # Aplicar transformación gamma: s = r^gamma
    corrected = np.power(image_float, gamma)
    
    # Si la entrada era uint8, convertir el resultado a uint8
    if image.dtype == np.uint8:
        corrected = (corrected * 255).astype(np.uint8)
    
    return corrected

def normalize_macenko(image, beta=0.15, alpha=1.0):
    """
    Implementa la normalización de Macenko para imágenes con tinción H&E.
    Esta función es útil para estandarizar imágenes biomédicas teñidas.
    
    Args:
        image: Imagen RGB de entrada (preferiblemente con tinción H&E)
        beta: Umbral para las componentes principales (default: 0.15)
        alpha: Factor de normalización (default: 1.0)
        
    Returns:
        Imagen normalizada según el método de Macenko
    """
    # Este método asume que la imagen tiene tinción H&E (hematoxilina y eosina)
    
    # Paso 1: Convertir a OD (densidad óptica)
    # OD = -log10(I), donde I es la intensidad normalizada [0,1]
    image_float = image.astype(np.float32) / 255.0
    # Evitar log(0)
    image_float[image_float == 0] = np.finfo(float).eps
    OD = -np.log10(image_float)
    
    # Paso 2: Remover valores extremadamente altos o bajos
    # Aplanar para procesar todos los píxeles a la vez
    OD_flat = OD.reshape((-1, 3))
    OD_thresh = OD_flat[(OD_flat > beta).any(axis=1), :]
    
    # Paso 3: SVD para encontrar los ejes principales de variación
    try:
        # Si no hay suficientes píxeles, puede fallar
        if OD_thresh.shape[0] < 2:
            raise ValueError("No hay suficientes píxeles para SVD")
            
        # SVD
        _, _, V = np.linalg.svd(OD_thresh, full_matrices=False)
        # Los vectores propios están en las filas de V
        # Nos quedamos con los dos primeros (H y E)
        vectors = V[:2].T
        
        # Paso 4: Proyectar en los vectores propios
        # Si los vectores están en la dirección incorrecta, invertirlos
        if vectors[0, 0] < 0:
            vectors[:, 0] = -vectors[:, 0]
        if vectors[0, 1] < 0:
            vectors[:, 1] = -vectors[:, 1]
        
        # Paso 5: Calcular coeficientes de proyección
        coeff = np.dot(OD_flat, vectors)
        
        # Paso 6: Reconstruir con coeficientes normalizados
        # Normalizar coeficientes
        OD_norm = np.zeros_like(OD_flat)
        for i in range(2):
            OD_norm += np.outer(coeff[:, i] * alpha, vectors[:, i])
        
        # Paso 7: Convertir de OD a RGB
        image_normalized = np.power(10, -OD_norm.reshape(OD.shape))
        # Recortar valores a [0,1]
        image_normalized = np.clip(image_normalized, 0, 1)
        
        # Convertir a uint8 si la entrada era uint8
        if image.dtype == np.uint8:
            image_normalized = (image_normalized * 255).astype(np.uint8)
        
        return image_normalized
        
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"Error en la normalización de Macenko: {str(e)}")
        # En caso de error, devolver la imagen original
        return image

def visualize_all_normalizations(image, title="Comparación de métodos de normalización"):
    """
    Visualiza una imagen con todas las técnicas de normalización implementadas.
    
    Args:
        image: Imagen de entrada
        title: Título de la figura
    """
    # Lista de métodos de normalización a aplicar
    methods = [
        ("Original", lambda x: x),
        ("Estadística (μ=0, σ=1)", normalize_statistical),
        ("Por canal (μ=0, σ=1)", normalize_per_channel),
        ("MinMax [0,1]", normalize_minmax),
        ("Ecualización Histograma", normalize_histogram_equalization),
        ("CLAHE", normalize_clahe),
        ("Gamma (0.7)", lambda x: normalize_gamma_correction(x, gamma=0.7)),
        ("Gamma (1.5)", lambda x: normalize_gamma_correction(x, gamma=1.5)),
        ("Macenko", normalize_macenko)
    ]
    
    # Crear figura
    n_methods = len(methods)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, (method_name, method_func) in enumerate(methods):
        # Aplicar normalización
        try:
            normalized = method_func(image)
            
            # Para visualización, asegurar que esté en [0,1] o [0,255]
            if normalized.dtype != np.uint8 and normalized.max() <= 1.0:
                # Está en [0,1], mantenerlo así
                pass
            elif normalized.dtype != np.uint8:
                # Está en otro rango, normalizarlo a [0,1]
                normalized = normalize_minmax(normalized)
        except Exception as e:
            print(f"Error al aplicar {method_name}: {str(e)}")
            normalized = image  # En caso de error, mostrar original
        
        # Mostrar imagen normalizada
        axes[i].imshow(normalized)
        axes[i].set_title(method_name)
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()
    
    return methods

def apply_normalization_batch(images, method='statistical'):
    """
    Aplica un método de normalización a un conjunto de imágenes.
    
    Args:
        images: Lista o array de imágenes
        method: Método de normalización a aplicar ('statistical', 'per_channel', 'minmax',
                'histogram_eq', 'clahe', 'gamma', 'macenko')
                
    Returns:
        Lista de imágenes normalizadas
    """
    method_map = {
        'statistical': normalize_statistical,
        'per_channel': normalize_per_channel,
        'minmax': normalize_minmax,
        'histogram_eq': normalize_histogram_equalization,
        'clahe': normalize_clahe,
        'gamma': lambda x: normalize_gamma_correction(x, gamma=0.7),
        'macenko': normalize_macenko
    }
    
    # Verificar que el método existe
    if method not in method_map:
        raise ValueError(f"Método de normalización '{method}' no reconocido")
    
    # Aplicar normalización a cada imagen
    normalized_images = []
    for img in images:
        normalized = method_map[method](img)
        normalized_images.append(normalized)
        
    return normalized_images

# Ejemplo de uso
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from skimage import io
    import os
    
    # Cargar una imagen de ejemplo
    sample_path = "../data/00001/image.png"
    if os.path.exists(sample_path):
        image = io.imread(sample_path)
        
        # Visualizar todas las normalizaciones
        visualize_all_normalizations(image, "Comparación de técnicas de normalización")
    else:
        print(f"No se encontró la imagen de ejemplo en {sample_path}")
