#!/usr/bin/env python3
"""
Dataset Analysis Script - DSB2018 Nuclei Segmentation
====================================================

Este script analiza el dataset de segmentación de núcleos para entender:
1. Estructura y organización de los datos
2. Dimensiones y propiedades de las imágenes
3. Características de las máscaras de segmentación
4. Variabilidad y patrones en el dataset

Ejecuta análisis exploratorio antes del desarrollo del pipeline.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from collections import defaultdict

def analyze_single_sample(sample_dir):
    """
    Analiza una muestra individual (carpeta con image.png y mask.png).
    
    Args:
        sample_dir (Path): Directorio de la muestra
        
    Returns:
        dict: Diccionario con propiedades de la muestra
    """
    image_path = sample_dir / "image.png"
    mask_path = sample_dir / "mask.png"
    
    # Verificar que los archivos existen
    if not (image_path.exists() and mask_path.exists()):
        return None
    
    # Cargar imagen original
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Cargar máscara
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    # Calcular propiedades
    analysis = {
        'sample_id': sample_dir.name,
        
        # Propiedades de imagen
        'image_height': image_rgb.shape[0],
        'image_width': image_rgb.shape[1],
        'image_channels': image_rgb.shape[2],
        'image_dtype': str(image_rgb.dtype),
        'image_min': image_rgb.min(),
        'image_max': image_rgb.max(),
        'image_mean': image_rgb.mean(),
        'image_std': image_rgb.std(),
        
        # Propiedades de máscara
        'mask_height': mask.shape[0],
        'mask_width': mask.shape[1],
        'mask_dtype': str(mask.dtype),
        'mask_min': mask.min(),
        'mask_max': mask.max(),
        'mask_unique_values': len(np.unique(mask)),
        'mask_unique_list': list(np.unique(mask)),
        
        # Análisis de contenido
        'mask_background_ratio': np.sum(mask == 0) / mask.size,
        'mask_foreground_ratio': np.sum(mask > 0) / mask.size,
        
        # Tamaños de archivo
        'image_file_size': image_path.stat().st_size,
        'mask_file_size': mask_path.stat().st_size,
    }
    
    return analysis, image_rgb, mask

def analyze_dataset_structure(data_dir, num_samples=10):
    """
    Analiza la estructura general del dataset.
    
    Args:
        data_dir (Path): Directorio del dataset
        num_samples (int): Número de muestras a analizar en detalle
        
    Returns:
        dict: Resumen del análisis del dataset
    """
    print("🔍 INICIANDO ANÁLISIS DEL DATASET")
    print("=" * 50)
    
    # Encontrar todas las carpetas de muestras
    sample_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    total_samples = len(sample_dirs)
    
    print(f"📊 Total de muestras encontradas: {total_samples}")
    
    # Seleccionar muestras para análisis detallado
    selected_samples = sorted(sample_dirs)[:num_samples]
    print(f"🎯 Analizando las primeras {len(selected_samples)} muestras en detalle...")
    
    # Análisis individual de muestras
    analyses = []
    images_for_display = []
    masks_for_display = []
    
    for i, sample_dir in enumerate(selected_samples):
        print(f"   📁 Procesando {sample_dir.name}...")
        
        result = analyze_single_sample(sample_dir)
        if result is not None:
            analysis, image, mask = result
            analyses.append(analysis)
            
            # Guardar las primeras 6 para visualización
            if len(images_for_display) < 6:
                images_for_display.append(image)
                masks_for_display.append(mask)
    
    # Crear DataFrame para análisis estadístico
    df = pd.DataFrame(analyses)
    
    print(f"\n✅ Análisis completado: {len(analyses)} muestras procesadas")
    
    return df, images_for_display, masks_for_display

def create_summary_report(df):
    """
    Crea un reporte resumen del análisis.
    
    Args:
        df (DataFrame): DataFrame con análisis de muestras
    """
    print("\n📋 REPORTE RESUMEN DEL DATASET")
    print("=" * 50)
    
    # Dimensiones de imágenes
    print("🖼️  DIMENSIONES DE IMÁGENES:")
    print(f"   Altura: {df['image_height'].describe()}")
    print(f"   Ancho: {df['image_width'].describe()}")
    print(f"   ¿Dimensiones constantes? {df['image_height'].nunique() == 1 and df['image_width'].nunique() == 1}")
    
    if df['image_height'].nunique() == 1 and df['image_width'].nunique() == 1:
        print(f"   📐 Tamaño estándar: {df['image_height'].iloc[0]} x {df['image_width'].iloc[0]}")
    else:
        print(f"   ⚠️  Tamaños variables detectados!")
    
    # Propiedades de máscaras
    print(f"\n🎯 PROPIEDADES DE MÁSCARAS:")
    print(f"   Valores únicos promedio: {df['mask_unique_values'].mean():.2f}")
    print(f"   Rango de valores únicos: {df['mask_unique_values'].min()} - {df['mask_unique_values'].max()}")
    print(f"   Ratio fondo promedio: {df['mask_background_ratio'].mean():.3f}")
    print(f"   Ratio núcleos promedio: {df['mask_foreground_ratio'].mean():.3f}")
    
    # Tamaños de archivo
    print(f"\n💾 TAMAÑOS DE ARCHIVO:")
    print(f"   Imagen promedio: {df['image_file_size'].mean()/1024:.1f} KB")
    print(f"   Máscara promedio: {df['mask_file_size'].mean()/1024:.1f} KB")
    
    # Tipos de datos
    print(f"\n🔢 TIPOS DE DATOS:")
    print(f"   Imagen: {df['image_dtype'].iloc[0]}")
    print(f"   Máscara: {df['mask_dtype'].iloc[0]}")
    print(f"   Rango valores imagen: {df['image_min'].min()} - {df['image_max'].max()}")
    print(f"   Rango valores máscara: {df['mask_min'].min()} - {df['mask_max'].max()}")

def create_visual_examples(images, masks, save_path):
    """
    Crea visualización de ejemplos del dataset.
    
    Args:
        images (list): Lista de imágenes
        masks (list): Lista de máscaras
        save_path (Path): Donde guardar la visualización
    """
    num_samples = min(6, len(images))
    
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 8))
    fig.suptitle('Ejemplos del Dataset: Imágenes Originales y Máscaras', fontsize=16, y=0.95)
    
    for i in range(num_samples):
        # Imagen original
        axes[0, i].imshow(images[i])
        axes[0, i].set_title(f'Imagen {i+1:05d}')
        axes[0, i].axis('off')
        
        # Máscara
        axes[1, i].imshow(masks[i], cmap='gray')
        axes[1, i].set_title(f'Máscara {i+1:05d}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Visualización guardada: {save_path}")
    plt.close()

def main():
    """Función principal del análisis."""
    print("🔬 ANÁLISIS EXPLORATORIO DEL DATASET DSB2018")
    print("=" * 60)
    
    # Configuración
    data_dir = Path("data")
    num_samples_to_analyze = 20
    
    # Verificar que existe el directorio
    if not data_dir.exists():
        print("❌ Error: Directorio 'data' no encontrado")
        print("💡 Asegúrate de que archive.zip fue extraído en 'data/'")
        return
    
    # Análisis del dataset
    try:
        df, images, masks = analyze_dataset_structure(data_dir, num_samples_to_analyze)
        
        # Reporte resumen
        create_summary_report(df)
        
        # Crear visualizaciones
        vis_path = Path("dataset_examples.png")
        create_visual_examples(images, masks, vis_path)
        
        # Guardar análisis detallado
        analysis_path = Path("dataset_analysis.csv")
        df.to_csv(analysis_path, index=False)
        print(f"💾 Análisis detallado guardado: {analysis_path}")
        
        print(f"\n🎉 ANÁLISIS COMPLETADO EXITOSAMENTE!")
        print(f"📁 Archivos generados:")
        print(f"   - {vis_path} (visualizaciones)")
        print(f"   - {analysis_path} (datos detallados)")
        
    except Exception as e:
        print(f"❌ Error durante el análisis: {e}")
        print("🔧 Verifica que las dependencias estén instaladas y los datos sean correctos")

if __name__ == "__main__":
    main()
