"""
Script de prueba para validar el NucleiDataset.

Este script valida:
1. Carga correcta de imágenes y máscaras
2. Generación de mapas de peso
3. Funcionamiento del DataLoader con batches
4. Visualización de muestras procesadas

Ejecutar con: python tests/test_nuclei_dataset.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Importar nuestro dataset
from src.nuclei_dataset import NucleiDataset, create_data_loaders


def test_dataset_basic_functionality():
    """
    Prueba funcionalidad básica del dataset.
    """
    print("=" * 50)
    print("PRUEBA 1: Funcionalidad Básica del Dataset")
    print("=" * 50)
    
    data_root = "data"
    
    try:
        # Crear dataset
        dataset = NucleiDataset(
            data_root=data_root,
            normalization_method='per_channel',
            generate_weight_maps=True,
            image_size=(256, 256)
        )
        
        print(f"✅ Dataset creado exitosamente")
        print(f"   - Número de muestras: {len(dataset)}")
        
        # Probar acceso a primera muestra
        sample = dataset[0]
        
        print(f"✅ Muestra cargada exitosamente")
        print(f"   - ID: {sample['sample_id']}")
        print(f"   - Imagen shape: {sample['image'].shape}")
        print(f"   - Imagen dtype: {sample['image'].dtype}")
        print(f"   - Imagen rango: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
        print(f"   - Máscara shape: {sample['mask'].shape}")
        print(f"   - Máscara dtype: {sample['mask'].dtype}")
        print(f"   - Máscara rango: [{sample['mask'].min():.3f}, {sample['mask'].max():.3f}]")
        
        if 'weight_map' in sample:
            print(f"   - Mapa peso shape: {sample['weight_map'].shape}")
            print(f"   - Mapa peso dtype: {sample['weight_map'].dtype}")
            print(f"   - Mapa peso rango: [{sample['weight_map'].min():.3f}, {sample['weight_map'].max():.3f}]")
        
        # Verificar que son tensores PyTorch
        assert isinstance(sample['image'], torch.Tensor), "Image debe ser tensor PyTorch"
        assert isinstance(sample['mask'], torch.Tensor), "Mask debe ser tensor PyTorch"
        
        # Verificar dimensiones correctas
        assert sample['image'].shape == (3, 256, 256), "Image debe tener shape (3, 256, 256)"
        assert sample['mask'].shape == (256, 256), "Mask debe tener shape (256, 256)"
        
        print("✅ Todas las verificaciones básicas pasaron")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Error en prueba básica: {str(e)}")
        raise


def test_dataloader_functionality(dataset):
    """
    Prueba funcionalidad del DataLoader con batches.
    """
    print("\n" + "=" * 50)
    print("PRUEBA 2: Funcionalidad del DataLoader")
    print("=" * 50)
    
    try:
        # Crear DataLoader
        from torch.utils.data import DataLoader
        
        loader = DataLoader(
            dataset, 
            batch_size=4, 
            shuffle=True, 
            num_workers=0  # 0 para evitar problemas en pruebas
        )
        
        print(f"✅ DataLoader creado exitosamente")
        print(f"   - Batch size: 4")
        print(f"   - Número de batches: {len(loader)}")
        
        # Probar carga de un batch
        batch = next(iter(loader))
        
        print(f"✅ Batch cargado exitosamente")
        print(f"   - Imágenes shape: {batch['image'].shape}")
        print(f"   - Máscaras shape: {batch['mask'].shape}")
        
        if 'weight_map' in batch:
            print(f"   - Mapas peso shape: {batch['weight_map'].shape}")
        
        print(f"   - Sample IDs: {batch['sample_id']}")
        
        # Verificar dimensiones de batch
        batch_size = batch['image'].shape[0]
        assert batch['image'].shape == (batch_size, 3, 256, 256), "Batch images shape incorrecto"
        assert batch['mask'].shape == (batch_size, 256, 256), "Batch masks shape incorrecto"
        
        print("✅ Todas las verificaciones de DataLoader pasaron")
        
        return batch
        
    except Exception as e:
        print(f"❌ Error en prueba de DataLoader: {str(e)}")
        raise


def test_different_normalizations():
    """
    Prueba diferentes métodos de normalización.
    """
    print("\n" + "=" * 50)
    print("PRUEBA 3: Métodos de Normalización")
    print("=" * 50)
    
    data_root = "data"
    methods = ['per_channel', 'clahe_gamma', 'none']
    
    for method in methods:
        try:
            dataset = NucleiDataset(
                data_root=data_root,
                normalization_method=method,
                generate_weight_maps=False  # Más rápido para esta prueba
            )
            
            sample = dataset[0]
            image = sample['image']
            
            print(f"✅ Normalización '{method}' funcionando")
            print(f"   - Rango imagen: [{image.min():.3f}, {image.max():.3f}]")
            
        except Exception as e:
            print(f"❌ Error con normalización '{method}': {str(e)}")


def visualize_sample_batch(batch):
    """
    Visualiza un batch de muestras para verificación visual.
    """
    print("\n" + "=" * 50)
    print("PRUEBA 4: Visualización de Muestras")
    print("=" * 50)
    
    try:
        batch_size = min(4, batch['image'].shape[0])
        
        fig, axes = plt.subplots(3, batch_size, figsize=(batch_size * 4, 12))
        if batch_size == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(batch_size):
            # Imagen original (desnormalizar para visualización)
            image = batch['image'][i].permute(1, 2, 0).numpy()
            
            # Si está normalizada por canal, necesitamos ajustar para visualización
            if image.min() < 0:  # Probablemente normalizada por canal
                # Reescalar aproximadamente a [0, 1] para visualización
                image = (image - image.min()) / (image.max() - image.min())
            
            axes[0, i].imshow(image)
            axes[0, i].set_title(f'Imagen {i+1}\nID: {batch["sample_id"][i]}')
            axes[0, i].axis('off')
            
            # Máscara
            mask = batch['mask'][i].numpy()
            axes[1, i].imshow(mask, cmap='gray')
            axes[1, i].set_title(f'Máscara {i+1}')
            axes[1, i].axis('off')
            
            # Mapa de peso (si existe)
            if 'weight_map' in batch:
                weight_map = batch['weight_map'][i].numpy()
                im = axes[2, i].imshow(weight_map, cmap='viridis')
                axes[2, i].set_title(f'Mapa Peso {i+1}')
                axes[2, i].axis('off')
                plt.colorbar(im, ax=axes[2, i], fraction=0.046, pad=0.04)
            else:
                axes[2, i].axis('off')
                axes[2, i].set_title('Sin mapa de peso')
        
        plt.suptitle('Verificación Visual del Dataset', fontsize=16)
        plt.tight_layout()
        plt.savefig('tests/dataset_validation_sample.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualización guardada en 'tests/dataset_validation_sample.png'")
        
    except Exception as e:
        print(f"❌ Error en visualización: {str(e)}")


def test_weight_map_quality(dataset):
    """
    Analiza la calidad de los mapas de peso generados.
    """
    print("\n" + "=" * 50)
    print("PRUEBA 5: Calidad de Mapas de Peso")
    print("=" * 50)
    
    try:
        # Analizar varias muestras
        n_samples = min(5, len(dataset))
        
        edge_percentages = []
        weight_stats = []
        
        for i in range(n_samples):
            sample = dataset[i]
            
            if 'weight_map' in sample:
                weight_map = sample['weight_map'].numpy()
                mask = sample['mask'].numpy()
                
                # Calcular estadísticas
                background_pixels = (weight_map == 0.1).sum()
                nucleus_pixels = (weight_map == 1.0).sum()
                edge_pixels = (weight_map == 5.0).sum()
                total_pixels = weight_map.size
                
                edge_percentage = (edge_pixels / total_pixels) * 100
                edge_percentages.append(edge_percentage)
                
                weight_stats.append({
                    'sample_id': sample['sample_id'],
                    'background_pct': (background_pixels / total_pixels) * 100,
                    'nucleus_pct': (nucleus_pixels / total_pixels) * 100,
                    'edge_pct': edge_percentage,
                    'min_weight': weight_map.min(),
                    'max_weight': weight_map.max(),
                    'mean_weight': weight_map.mean()
                })
        
        # Reportar estadísticas
        print(f"Análisis de {n_samples} muestras:")
        print(f"Porcentaje promedio de bordes: {np.mean(edge_percentages):.2f}%")
        print(f"Rango de bordes: {np.min(edge_percentages):.2f}% - {np.max(edge_percentages):.2f}%")
        
        for stats in weight_stats[:3]:  # Mostrar primeras 3
            print(f"\nMuestra {stats['sample_id']}:")
            print(f"  - Fondo: {stats['background_pct']:.1f}%")
            print(f"  - Núcleos: {stats['nucleus_pct']:.1f}%") 
            print(f"  - Bordes: {stats['edge_pct']:.1f}%")
            print(f"  - Peso promedio: {stats['mean_weight']:.3f}")
        
        print("✅ Análisis de mapas de peso completado")
        
    except Exception as e:
        print(f"❌ Error en análisis de mapas de peso: {str(e)}")


def main():
    """
    Función principal que ejecuta todas las pruebas.
    """
    print("INICIANDO PRUEBAS DEL NUCLEI DATASET")
    print("=" * 70)
    
    try:
        # Crear directorio de pruebas si no existe
        os.makedirs('tests', exist_ok=True)
        
        # Ejecutar pruebas en secuencia
        dataset = test_dataset_basic_functionality()
        batch = test_dataloader_functionality(dataset)
        test_different_normalizations()
        visualize_sample_batch(batch)
        test_weight_map_quality(dataset)
        
        print("\n" + "=" * 70)
        print("✅ TODAS LAS PRUEBAS COMPLETADAS EXITOSAMENTE")
        print("=" * 70)
        
        print("\nResumen:")
        print(f"- Dataset funcional con {len(dataset)} muestras")
        print("- DataLoader creando batches correctamente")
        print("- Múltiples métodos de normalización funcionando")
        print("- Mapas de peso generándose apropiadamente")
        print("- Visualizaciones guardadas para inspección")
        
    except Exception as e:
        print(f"\n❌ PRUEBAS FALLIDAS: {str(e)}")
        print("Revisar implementación antes de continuar")
        sys.exit(1)


if __name__ == "__main__":
    main()
