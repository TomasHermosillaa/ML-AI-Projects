"""
Script de Inferencia para U-Net de Segmentación de Núcleos.

Este script permite cargar el modelo entrenado y hacer predicciones
sobre imágenes nuevas, mostrando visualizaciones de los resultados.

Autor: Proyecto de Segmentación de Núcleos
Fecha: 2025
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional
from scipy import ndimage
from scipy.ndimage import maximum_filter
from skimage.segmentation import watershed
from skimage.measure import label, regionprops

from .conv_blocks import BasicUNet
from .normalization import normalize_per_channel


def find_local_maxima(image, min_distance=10, threshold_abs=0.3):
    """
    Encuentra máximos locales en una imagen.
    
    Args:
        image: Imagen 2D
        min_distance: Distancia mínima entre máximos
        threshold_abs: Umbral absoluto mínimo
        
    Returns:
        Array de coordenadas (N, 2) de máximos locales
    """
    # Aplicar filtro de máximo local
    local_max = maximum_filter(image, size=min_distance) == image
    
    # Aplicar umbral
    above_threshold = image > threshold_abs
    
    # Combinar condiciones
    local_maxima = local_max & above_threshold
    
    # Obtener coordenadas
    coords = np.argwhere(local_maxima)
    
    return coords


class NucleiPredictor:
    """Clase para hacer inferencia con el modelo U-Net entrenado."""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Inicializa el predictor.
        
        Args:
            model_path: Path al archivo .pth del modelo entrenado
            device: Dispositivo ('cuda', 'cpu', o 'auto')
        """
        self.device = torch.device(
            'cuda' if device == 'auto' and torch.cuda.is_available() 
            else device if device != 'auto' else 'cpu'
        )
        
        # Inicializar el modelo
        self.model = BasicUNet(in_channels=3, out_channels=1)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Modelo cargado desde: {model_path}")
        print(f"Dispositivo: {self.device}")
    
    def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
        """
        Preprocesa una imagen para inferencia.
        
        Args:
            image: Imagen RGB como numpy array (H, W, 3)
            target_size: Tamaño objetivo (width, height)
            
        Returns:
            Tensor preprocessado (1, 3, H, W)
        """
        # Redimensionar si es necesario
        if image.shape[:2] != target_size[::-1]:  # target_size es (W, H), shape es (H, W)
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalizar usando el mismo método que en entrenamiento
        image_norm = normalize_per_channel(image)
        
        # Convertir a tensor y agregar dimensiones de batch
        image_tensor = torch.from_numpy(np.transpose(image_norm, (2, 0, 1))).float()
        image_tensor = image_tensor.unsqueeze(0)  # Agregar dimensión de batch
        
        return image_tensor
    
    def postprocess_prediction(self, prediction: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
        """
        Postprocesa la predicción del modelo.
        
        Args:
            prediction: Tensor de predicción (1, 1, H, W)
            threshold: Umbral para binarización
            
        Returns:
            Máscara binaria como numpy array (H, W)
        """
        # Aplicar sigmoid y quitar dimensiones extra
        pred_sigmoid = torch.sigmoid(prediction).squeeze().cpu().numpy()
        
        # Binarizar
        binary_mask = (pred_sigmoid > threshold).astype(np.uint8)
        
        return binary_mask
    
    def predict(self, image: np.ndarray, threshold: float = 0.5) -> dict:
        """
        Hace una predicción completa sobre una imagen.
        
        Args:
            image: Imagen RGB como numpy array (H, W, 3)
            threshold: Umbral para binarización
            
        Returns:
            Dict con 'raw_prediction', 'binary_mask', 'preprocessed_image'
        """
        original_size = image.shape[:2]
        
        # Preprocesar
        image_tensor = self.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)
        
        # Inferencia
        with torch.no_grad():
            raw_prediction = self.model(image_tensor)
        
        # Postprocesar
        binary_mask = self.postprocess_prediction(raw_prediction, threshold)
        
        # Redimensionar al tamaño original si es necesario
        if binary_mask.shape != original_size:
            binary_mask = cv2.resize(
                binary_mask.astype(np.uint8), 
                (original_size[1], original_size[0]), 
                interpolation=cv2.INTER_NEAREST
            )
        
        return {
            'raw_prediction': torch.sigmoid(raw_prediction).squeeze().cpu().numpy(),
            'binary_mask': binary_mask,
            'preprocessed_image': image_tensor.squeeze().cpu().numpy()
        }
    
    def predict_from_path(self, image_path: str, threshold: float = 0.5) -> dict:
        """
        Hace predicción desde un archivo de imagen.
        
        Args:
            image_path: Path a la imagen
            threshold: Umbral para binarización
            
        Returns:
            Dict con resultados de predicción
        """
        # Cargar imagen
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        # Convertir BGR a RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return self.predict(image, threshold)

    def count_cells(self, binary_mask: np.ndarray, method: str = 'watershed') -> dict:
        """
        Cuenta las células en la máscara binaria.
        
        Args:
            binary_mask: Máscara binaria (H, W)
            method: Método de conteo ('simple', 'watershed', 'distance')
            
        Returns:
            Dict con conteo y información adicional
        """
        if method == 'simple':
            return self._count_simple(binary_mask)
        elif method == 'watershed':
            return self._count_watershed(binary_mask)
        elif method == 'distance':
            return self._count_distance_transform(binary_mask)
        else:
            raise ValueError(f"Método no soportado: {method}")
    
    def _count_simple(self, binary_mask: np.ndarray) -> dict:
        """Conteo simple usando componentes conectados."""
        labeled_mask = label(binary_mask)
        num_cells = labeled_mask.max()
        
        return {
            'count': num_cells,
            'method': 'simple',
            'labeled_mask': labeled_mask,
            'cell_properties': regionprops(labeled_mask)
        }
    
    def _count_watershed(self, binary_mask: np.ndarray) -> dict:
        """Conteo usando watershed para separar células tocándose."""
        # Calcular distancia transform
        distance = ndimage.distance_transform_edt(binary_mask)
        
        # Encontrar máximos locales (centros de células) - parámetros más permisivos
        local_maxima = find_local_maxima(distance, min_distance=5, threshold_abs=1.0)
        markers = np.zeros(distance.shape, dtype=bool)
        if len(local_maxima) > 0:
            markers[local_maxima[:, 0], local_maxima[:, 1]] = True
        markers = label(markers)[0]
        
        # Aplicar watershed
        labels = watershed(-distance, markers, mask=binary_mask)
        
        num_cells = labels.max()
        
        return {
            'count': num_cells,
            'method': 'watershed',
            'labeled_mask': labels,
            'distance_transform': distance,
            'markers': markers,
            'cell_properties': regionprops(labels)
        }
    
    def _count_distance_transform(self, binary_mask: np.ndarray) -> dict:
        """Conteo usando análisis de distancia transform."""
        # Calcular distancia transform
        distance = ndimage.distance_transform_edt(binary_mask)
        
        # Encontrar máximos locales con parámetros adaptativos
        local_maxima = find_local_maxima(distance, min_distance=5, threshold_abs=1.0)
        
        # Filtrar máximos por intensidad mínima
        valid_peaks = []
        for peak in local_maxima:
            if distance[peak[0], peak[1]] > 2.0:  # Mínimo radio de 2 píxeles
                valid_peaks.append(peak)
        
        num_cells = len(valid_peaks)
        
        # Crear máscara de centros
        centers_mask = np.zeros(binary_mask.shape, dtype=bool)
        if valid_peaks:
            centers_coords = np.array(valid_peaks)
            centers_mask[centers_coords[:, 0], centers_coords[:, 1]] = True
        
        return {
            'count': num_cells,
            'method': 'distance_transform',
            'distance_transform': distance,
            'cell_centers': valid_peaks,
            'centers_mask': centers_mask
        }


def visualize_counting_results(image: np.ndarray, 
                             prediction_result: dict,
                             counting_result: dict,
                             ground_truth: Optional[np.ndarray] = None,
                             save_path: Optional[str] = None) -> None:
    """
    Visualiza los resultados de predicción y conteo.
    
    Args:
        image: Imagen original RGB
        prediction_result: Resultado de predict()
        counting_result: Resultado de count_cells()
        ground_truth: Máscara verdadera opcional
        save_path: Path para guardar la visualización
    """
    n_plots = 5 if ground_truth is None else 6
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Imagen original
    axes[0].imshow(image)
    axes[0].set_title('Imagen Original')
    axes[0].axis('off')
    
    # Predicción cruda
    im1 = axes[1].imshow(prediction_result['raw_prediction'], cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title('Predicción (Probabilidades)')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # Máscara binaria
    axes[2].imshow(prediction_result['binary_mask'], cmap='gray')
    axes[2].set_title('Máscara Predicha')
    axes[2].axis('off')
    
    # Células etiquetadas
    labeled_mask = counting_result['labeled_mask']
    axes[3].imshow(labeled_mask, cmap='nipy_spectral')
    axes[3].set_title(f'Células Identificadas\nConteo: {counting_result["count"]} células')
    axes[3].axis('off')
    
    # Visualización con centros o distancia
    if 'distance_transform' in counting_result:
        axes[4].imshow(counting_result['distance_transform'], cmap='hot')
        axes[4].set_title('Transform de Distancia')
        if 'cell_centers' in counting_result:
            centers = counting_result['cell_centers']
            if centers:
                centers_array = np.array(centers)
                axes[4].scatter(centers_array[:, 1], centers_array[:, 0], c='cyan', s=20, marker='x')
    else:
        axes[4].imshow(labeled_mask > 0, cmap='gray')
        axes[4].set_title('Máscara de Células')
    axes[4].axis('off')
    
    # Ground truth (si está disponible)
    if ground_truth is not None:
        axes[5].imshow(ground_truth, cmap='gray')
        axes[5].set_title('Ground Truth')
        axes[5].axis('off')
    else:
        # Overlay de células sobre imagen original
        overlay = image.copy()
        if 'cell_centers' in counting_result and counting_result['cell_centers']:
            centers = np.array(counting_result['cell_centers'])
            for center in centers:
                cv2.circle(overlay, (int(center[1]), int(center[0])), 5, (255, 0, 0), 2)
        axes[5].imshow(overlay)
        axes[5].set_title(f'Células Detectadas\n({counting_result["method"]})')
        axes[5].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualización guardada en: {save_path}")
    
    plt.close()  # Cerrar para evitar mostrar ventana


def visualize_prediction(image: np.ndarray, 
                        prediction_result: dict, 
                        ground_truth: Optional[np.ndarray] = None,
                        save_path: Optional[str] = None) -> None:
    """
    Visualiza los resultados de predicción.
    
    Args:
        image: Imagen original RGB
        prediction_result: Resultado de predict()
        ground_truth: Máscara verdadera opcional (para comparación)
        save_path: Path para guardar la visualización
    """
    n_plots = 3 if ground_truth is None else 4
    fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 4))
    
    # Imagen original
    axes[0].imshow(image)
    axes[0].set_title('Imagen Original')
    axes[0].axis('off')
    
    # Predicción cruda (mapa de probabilidades)
    im1 = axes[1].imshow(prediction_result['raw_prediction'], cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title('Predicción (Probabilidades)')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # Máscara binaria
    axes[2].imshow(prediction_result['binary_mask'], cmap='gray')
    axes[2].set_title('Máscara Predicha')
    axes[2].axis('off')
    
    # Ground truth (si está disponible)
    if ground_truth is not None:
        axes[3].imshow(ground_truth, cmap='gray')
        axes[3].set_title('Ground Truth')
        axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualización guardada en: {save_path}")
    
    plt.show()


def calculate_metrics(predicted_mask: np.ndarray, 
                     ground_truth_mask: np.ndarray) -> dict:
    """
    Calcula métricas de evaluación entre predicción y ground truth.
    
    Args:
        predicted_mask: Máscara predicha (binaria)
        ground_truth_mask: Máscara verdadera (binaria)
        
    Returns:
        Dict con métricas calculadas
    """
    # Binarizar máscaras
    pred_bin = (predicted_mask > 0).astype(bool)
    gt_bin = (ground_truth_mask > 0).astype(bool)
    
    # Calcular métricas
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    
    # Dice Score
    dice = (2.0 * intersection) / (pred_bin.sum() + gt_bin.sum()) if (pred_bin.sum() + gt_bin.sum()) > 0 else 1.0
    
    # IoU (Intersection over Union)
    iou = intersection / union if union > 0 else 1.0
    
    # Precision y Recall
    true_positive = intersection
    false_positive = pred_bin.sum() - true_positive
    false_negative = gt_bin.sum() - true_positive
    
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 1.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 1.0
    
    return {
        'dice_score': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'intersection': intersection,
        'union': union
    }


if __name__ == "__main__":
    # Ejemplo de uso
    model_path = "/home/tomy/Documents/ML-AI-Projects/unet_nuclei.pth"
    test_image_path = "/home/tomy/Documents/ML-AI-Projects/data/00001/image.png"
    test_mask_path = "/home/tomy/Documents/ML-AI-Projects/data/00001/mask.png"
    
    # Crear predictor
    predictor = NucleiPredictor(model_path)
    
    # Hacer predicción
    result = predictor.predict_from_path(test_image_path)
    
    # Cargar imagen y ground truth para visualización
    image = cv2.imread(test_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    ground_truth = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Calcular métricas de segmentación
    metrics = calculate_metrics(result['binary_mask'], ground_truth)
    print("\n--- Métricas de Segmentación ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # ¡CONTEO DE CÉLULAS!
    print("\n" + "="*50)
    print("🔬 CONTEO DE CÉLULAS")
    print("="*50)
    
    # Probar diferentes métodos de conteo
    methods = ['simple', 'watershed', 'distance']
    counting_results = {}
    
    for method in methods:
        count_result = predictor.count_cells(result['binary_mask'], method=method)
        counting_results[method] = count_result
        print(f"\n📊 Método '{method.upper()}':")
        print(f"   Células detectadas: {count_result['count']}")
        
        # Información adicional por método
        if method == 'watershed' or method == 'distance':
            if 'cell_centers' in count_result:
                print(f"   Centros identificados: {len(count_result['cell_centers'])}")
        
        # Estadísticas de tamaño de células (si disponible)
        if 'cell_properties' in count_result:
            areas = [prop.area for prop in count_result['cell_properties']]
            if areas:
                print(f"   Área promedio: {np.mean(areas):.1f} píxeles")
                print(f"   Área min/max: {min(areas):.0f}/{max(areas):.0f} píxeles")
    
    # Usar el método simple como principal (más preciso en este caso)
    best_result = counting_results['simple']
    
    print(f"\n🎯 RESULTADO FINAL: {best_result['count']} células detectadas")
    print(f"   (Método: {best_result['method']})")
    
    # Visualizar resultados de conteo
    output_path = "/home/tomy/Documents/ML-AI-Projects/cell_counting_result.png"
    visualize_counting_results(image, result, best_result, ground_truth, save_path=output_path)
    
    # Comparar con ground truth si es posible
    gt_count = predictor.count_cells((ground_truth > 127).astype(np.uint8), method='simple')
    print(f"\n📋 Comparación con Ground Truth:")
    print(f"   Ground Truth: {gt_count['count']} células")
    print(f"   Predicción: {best_result['count']} células") 
    print(f"   Diferencia: {abs(best_result['count'] - gt_count['count'])} células")
    
    if gt_count['count'] > 0:
        accuracy = min(best_result['count'], gt_count['count']) / max(best_result['count'], gt_count['count'])
        print(f"   Precisión de conteo: {accuracy:.1%}")
