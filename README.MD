
# 🔬 Segmentación de Núcleos Celulares - Proyecto ML End-to-End
Tomás Hermosilla - 2025
Proyecto de **segmentación de instancias de núcleos celulares** usando **Deep Learning** con arquitectura U-Net implementada desde cero. Incluye pipeline completo de ingeniería de datos, entrenamiento y despliegue.

## 🎯 Objetivo del Proyecto

Desarrollar un sistema completo de segmentación que pueda:
- Identificar y delimitar núcleos individuales en imágenes microscópicas
- Separar núcleos que se tocan usando mapas de peso morfológicos
- Procesar imágenes biomédicas con diferentes técnicas de tinción
- Proporcionar máscaras precisas para análisis posterior

---

## 📁 Estructura del Proyecto

```
ML-AI-Projects/
├── data/                           # Dataset DSB2018 (2,224 muestras)
│   ├── 00001/
│   │   ├── image.png              # Imagen RGB microscópica
│   │   └── mask.png               # Máscara binaria consolidada
│   ├── 00002/
│   └── ...
├── src/                           # Código fuente principal
│   ├── nuclei_dataset.py          # Dataset PyTorch personalizado
│   ├── normalization.py           # Técnicas de normalización biomédica
│   └── weight_maps.py            # Generación de mapas de peso morfológicos
├── notebooks/                     # Análisis y experimentación
│   ├── exploratory_analysis.ipynb     # EDA del dataset
│   ├── morphological_operations.ipynb # Operaciones morfológicas
│   ├── normalization_techniques.ipynb # Comparación de normalizaciones
│   └── weight_maps_analysis.ipynb     # Análisis de mapas de peso
├── tests/                         # Scripts de prueba y validación
│   ├── test_nuclei_dataset.py     # Validación completa del dataset
│   ├── verify_installation.py     # Verificación del entorno
│   └── analyze_dataset.py         # Análisis estadístico del dataset
├── plan/                          # Documentación del proyecto
│   ├── Plan_Detallado_Segmentacion_Nucleos.md
│   ├── analisis_exploratorio_dataset.md
│   ├── analisis_normalizacion_imagenes.md
│   └── analisis_mapas_peso.md
├── examples/                      # Ejemplos y visualizaciones
│   ├── dataset_examples.png
│   └── dataset_analysis.csv
├── images/                        # Imágenes generadas por análisis
│   ├── dataset_validation_sample.png
│   ├── image-stats.png
│   └── image-weight.png
├── venv/                          # Entorno virtual Python
├── requirements.txt               # Dependencias del proyecto
└── README.md                      # Este archivo
```

---

## 🛠️ Configuración del Entorno

### 1. Prerrequisitos
- **Python 3.8+**
- **CUDA 12.1+** (para GPU)
- **8GB+ VRAM** recomendado

### 2. Instalación Paso a Paso

```bash
# 1. Clonar y acceder al proyecto
cd /path/to/ML-AI-Projects

# 2. Crear entorno virtual
python -m venv venv

# 3. Activar entorno virtual
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 4. Verificar CUDA (opcional pero recomendado)
nvidia-smi

# 5. Instalar PyTorch con soporte CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 6. Instalar dependencias adicionales
pip install -r requirements.txt

# 7. Verificar instalación
python tests/verify_installation.py
```

### 3. Configurar Jupyter (Opcional)
```bash
python -m ipykernel install --user --name=nuclei_segmentation --display-name="Nuclei Segmentation"
jupyter lab --no-browser --port=8888
```

---

## 📚 Dataset (DSB2018)

### Preparación
1. **Descargar** dataset desde Kaggle: `2018 Data Science Bowl`
2. **Descomprimir** en carpeta `data/`
3. **Verificar** estructura:

```bash
python tests/analyze_dataset.py
```

### Características del Dataset
- **📊 2,224 muestras** - Robusto para entrenamiento
- **🖼️ Dimensiones fijas**: 256×256 píxeles
- **🎯 Máscaras binarias**: Núcleos consolidados (0=fondo, 255=núcleo)
- **⚖️ Desbalance de clases**: 12.2% núcleos vs 87.8% fondo
- **🔬 Variedad**: Diferentes técnicas de tinción y tipos celulares

---

## 🔧 Componentes Implementados

### 📊 **Etapa 1: Análisis Exploratorio** ✅
- **EDA completo** del dataset DSB2018
- **Visualizaciones** estadísticas de distribuciones
- **Identificación** de patrones y desafíos
- **Análisis** de calidad de máscaras

### 🛠️ **Etapa 2: Ingeniería de Datos** ✅

#### Técnicas de Normalización (`src/normalization.py`)
```python
from src.normalization import normalize_per_channel, normalize_clahe

# Normalización por canal (recomendada)
normalized = normalize_per_channel(image)

# CLAHE + Corrección Gamma para contraste
enhanced = normalize_clahe(image, clip_limit=2.0)
```

#### Mapas de Peso Morfológicos (`src/weight_maps.py`)
```python
from src.weight_maps import create_edge_weight_map

# Generar mapa que enfatiza bordes entre núcleos
weight_map, edges = create_edge_weight_map(mask, edge_weight=5.0)
```

#### Dataset PyTorch (`src/nuclei_dataset.py`)
```python
from src.nuclei_dataset import NucleiDataset
from torch.utils.data import DataLoader

# Crear dataset con configuración avanzada
dataset = NucleiDataset(
    data_root="data",
    normalization_method='per_channel',
    generate_weight_maps=True,
    image_size=(256, 256)
)

# DataLoader para entrenamiento
loader = DataLoader(dataset, batch_size=4, shuffle=True)
```

---

## 🧪 Pruebas y Validación

### Ejecutar Todas las Pruebas
```bash
# Prueba completa del dataset (recomendado)
python tests/test_nuclei_dataset.py

# Análisis del dataset
python tests/analyze_dataset.py

# Verificar instalaciones
python tests/verify_installation.py
```

### Explorar con Notebooks
```bash
jupyter lab

# Navegar a:
# - notebooks/exploratory_analysis.ipynb
# - notebooks/morphological_operations.ipynb  
# - notebooks/normalization_techniques.ipynb
```

---

## 📈 Estado Actual del Desarrollo

### ✅ **Completado**
- [x] **Análisis exploratorio** completo del dataset
- [x] **6 técnicas de normalización** implementadas y comparadas
- [x] **Mapas de peso morfológicos** con gradiente morfológico
- [x] **Dataset PyTorch personalizado** con lazy loading
- [x] **Pipeline de datos completo** validado con 2,224 muestras
- [x] **Pruebas automatizadas** con cobertura del 100%

### 🔄 **En Progreso**
- [ ] Arquitectura U-Net desde cero
- [ ] Bloques convolucionales modulares
- [ ] Skip connections optimizadas

### 📋 **Próximos Pasos**
1. **Arquitectura Neural Básica** (Etapa 3)
2. **Función de Pérdida DiceBCE** (Etapa 4)
3. **Loop de Entrenamiento** con AMP (Etapa 5)
4. **Post-procesamiento Watershed** (Etapa 6)

---

## 🎯 Especificaciones Técnicas

### Tensores del Dataset
```python
# Salida típica de NucleiDataset[i]
sample = {
    'image': torch.Tensor,      # Shape: (3, 256, 256), dtype: float32
    'mask': torch.Tensor,       # Shape: (256, 256), dtype: float32, rango: [0, 1]  
    'weight_map': torch.Tensor, # Shape: (256, 256), dtype: float32, rango: [0.1, 5.0]
    'sample_id': str            # ID de la muestra: "00001", "00002", etc.
}

# Batch del DataLoader
batch = {
    'image': torch.Tensor,      # Shape: (B, 3, 256, 256)
    'mask': torch.Tensor,       # Shape: (B, 256, 256)
    'weight_map': torch.Tensor, # Shape: (B, 256, 256)
    'sample_id': List[str]      # Lista de IDs del batch
}
```

### Mapas de Peso
- **Fondo**: 0.1 (peso bajo)
- **Núcleos**: 1.0 (peso normal)
- **Bordes**: 5.0 (peso alto - énfasis en separación)
- **Promedio de bordes**: ~6.18% de píxeles por imagen

### Métodos de Normalización
- **`per_channel`**: Media=0, STD=1 por canal RGB ⭐ **Recomendada**
- **`clahe_gamma`**: CLAHE + corrección gamma para contraste
- **`none`**: Escalado simple a [0, 1]

---

## 🚀 Uso Rápido

```python
# Ejemplo completo de uso del dataset
from src.nuclei_dataset import NucleiDataset, create_data_loaders
import matplotlib.pyplot as plt

# 1. Crear dataset
dataset = NucleiDataset("data", generate_weight_maps=True)
print(f"Dataset cargado: {len(dataset)} muestras")

# 2. Obtener una muestra
sample = dataset[0]
print(f"Imagen: {sample['image'].shape}")
print(f"Máscara: {sample['mask'].shape}")  
print(f"Pesos: {sample['weight_map'].shape}")

# 3. Crear DataLoaders
train_loader, val_loader = create_data_loaders("data", batch_size=4)

# 4. Iterar sobre batches
for batch in train_loader:
    images = batch['image']      # (4, 3, 256, 256)
    masks = batch['mask']        # (4, 256, 256)
    weights = batch['weight_map'] # (4, 256, 256)
    break
```

---

## 📞 Información de Contacto

**Autor**: Tomás Hermosilla
**Contacto**: tomas.hermosilla.p@gmail.com 

---

## 📜 Licencia

Este proyecto abierto es de propósito educativo y de investigación.