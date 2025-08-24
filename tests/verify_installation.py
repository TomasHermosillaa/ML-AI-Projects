#!/usr/bin/env python3
"""
Script de Verificación de Instalación - Proyecto Segmentación de Núcleos
Verifica que todas las dependencias estén instaladas y funcionando correctamente
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_pytorch():
    """Verificar instalación de PyTorch y soporte CUDA"""
    print("🔥 Verificando PyTorch...")
    try:
        import torch
        import torchvision
        
        print(f"   ✅ PyTorch version: {torch.__version__}")
        print(f"   ✅ TorchVision version: {torchvision.__version__}")
        
        # Verificar CUDA
        if torch.cuda.is_available():
            print(f"   🚀 CUDA disponible: {torch.version.cuda}")
            print(f"   🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"   💾 VRAM disponible: {torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB")
            
            # Test básico con GPU
            x = torch.rand(2, 3, 256, 256).cuda()
            print(f"   ✅ Test tensor GPU: {x.shape} en {x.device}")
            del x
            torch.cuda.empty_cache()
        else:
            print("   ⚠️  CUDA no disponible - usando CPU")
        
        return True
    except Exception as e:
        print(f"   ❌ Error PyTorch: {e}")
        return False

def test_opencv():
    """Verificar instalación de OpenCV"""
    print("\n📸 Verificando OpenCV...")
    try:
        import cv2
        import numpy as np
        
        print(f"   ✅ OpenCV version: {cv2.__version__}")
        
        # Test básico de operaciones
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        print(f"   ✅ Test imagen: {img.shape} -> {gray.shape}")
        print("   ✅ Operaciones morfológicas disponibles")
        
        return True
    except Exception as e:
        print(f"   ❌ Error OpenCV: {e}")
        return False

def test_scientific_libs():
    """Verificar bibliotecas científicas"""
    print("\n🧮 Verificando bibliotecas científicas...")
    try:
        import numpy as np
        import scipy
        from scipy import ndimage
        
        print(f"   ✅ NumPy version: {np.__version__}")
        print(f"   ✅ SciPy version: {scipy.__version__}")
        
        # Test básico
        arr = np.random.rand(100, 100)
        filtered = ndimage.gaussian_filter(arr, sigma=1.0)
        
        print(f"   ✅ Test arrays: {arr.shape} -> {filtered.shape}")
        
        return True
    except Exception as e:
        print(f"   ❌ Error bibliotecas científicas: {e}")
        return False

def test_visualization():
    """Verificar bibliotecas de visualización"""
    print("\n📊 Verificando visualización...")
    try:
        import matplotlib
        matplotlib.use('Agg')  # Backend sin GUI para testing
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        print(f"   ✅ Matplotlib version: {matplotlib.__version__}")
        print(f"   ✅ Seaborn version: {sns.__version__}")
        
        # Test básico
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        ax.plot([1, 2, 3], [1, 4, 2])
        plt.close(fig)
        
        print("   ✅ Test plotting exitoso")
        
        return True
    except Exception as e:
        print(f"   ❌ Error visualización: {e}")
        return False

def test_utilities():
    """Verificar utilidades"""
    print("\n🛠️ Verificando utilidades...")
    try:
        import tqdm
        from PIL import Image
        
        print(f"   ✅ tqdm version: {tqdm.__version__}")
        print(f"   ✅ Pillow disponible")
        
        # Test progress bar
        from tqdm import tqdm as progress_bar
        import time
        
        for i in progress_bar(range(3), desc="   Test progress"):
            time.sleep(0.1)
        
        return True
    except Exception as e:
        print(f"   ❌ Error utilidades: {e}")
        return False

def test_jupyter():
    """Verificar Jupyter Lab"""
    print("\n📓 Verificando Jupyter...")
    try:
        import jupyter_core
        import jupyterlab
        import ipykernel
        
        print(f"   ✅ Jupyter Core version: {jupyter_core.__version__}")
        print("   ✅ JupyterLab instalado")
        print("   ✅ IPython kernel disponible")
        
        return True
    except Exception as e:
        print(f"   ❌ Error Jupyter: {e}")
        return False

def create_requirements():
    """Crear archivo requirements.txt"""
    print("\n📄 Creando requirements.txt...")
    try:
        requirements = [
            "torch>=2.0.0",
            "torchvision>=0.15.0", 
            "opencv-python>=4.8.0",
            "numpy>=1.24.0",
            "scipy>=1.10.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "pillow>=10.0.0",
            "tqdm>=4.65.0",
            "jupyterlab>=4.0.0",
            "ipykernel>=6.25.0"
        ]
        
        with open("requirements.txt", "w") as f:
            f.write("\n".join(requirements))
        
        print("   ✅ requirements.txt creado")
        return True
    except Exception as e:
        print(f"   ❌ Error creando requirements: {e}")
        return False

def main():
    """Ejecutar todas las verificaciones"""
    print("🚀 VERIFICACIÓN DE INSTALACIÓN - PROYECTO SEGMENTACIÓN DE NÚCLEOS")
    print("=" * 65)
    
    tests = [
        test_pytorch,
        test_opencv, 
        test_scientific_libs,
        test_visualization,
        test_utilities,
        test_jupyter
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # Crear requirements.txt
    create_requirements()
    
    print("\n" + "=" * 65)
    print("📋 RESUMEN DE VERIFICACIÓN:")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"   🎉 ¡TODAS LAS VERIFICACIONES EXITOSAS! ({passed}/{total})")
        print("   ✅ El entorno está listo para el proyecto")
        print("\n🎯 SIGUIENTE PASO: Activar Jupyter Lab con:")
        print("   jupyter lab")
        return True
    else:
        print(f"   ⚠️  Verificaciones: {passed}/{total} exitosas")
        print("   ❌ Hay problemas que resolver antes de continuar")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
