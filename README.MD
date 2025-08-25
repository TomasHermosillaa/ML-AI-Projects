
# Segmentación y Conteo de Núcleos Celulares - Proyecto ML End-to-End
Tomás Hermosilla - 2025

Sistema completo de **segmentación y conteo automático de células** usando **U-Net** desde cero.

## Objetivo del Proyecto COMPLETADO

Sistema completo de segmentación y conteo que:
- ✅ **Identifica y delimita núcleos individuales** en imágenes microscópicas
- ✅ **Cuenta células automáticamente** con 78.3% de precisión
- ✅ **Segmenta con 89.82% Dice Score** y 95.31% de precisión 
- ✅ **Procesa imágenes biomédicas** con técnicas de normalización avanzadas
- ✅ **Pipeline end-to-end funcional**: Imagen → Segmentación → Conteo

## Resultados Principales

### Rendimiento del Modelo
- **Dice Score**: 89.82% (excelente precisión de segmentación)
- **IoU**: 81.53% (alta superposición con ground truth)
- **Precision**: 95.31% (muy pocas falsas detecciones)
- **Recall**: 84.93% (detecta la mayoría de núcleos)

### Conteo de Células
- **Método óptimo**: Componentes conectados simples
- **Precisión de conteo**: 78.3% (18/23 células detectadas)
- **Error promedio**: ±5 células por imagen
- **Métodos implementados**: Simple, Watershed, Distance Transform

---

## Estructura del Proyecto

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
│   ├── conv_blocks.py             # Arquitectura U-Net desde cero
│   ├── train.py                   # Script de entrenamiento completo
│   └── inference.py               # Predicción y conteo de células
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
├── unet_nuclei.pth                # Modelo entrenado listo para uso
├── cell_counting_result.png       # Visualización de conteo de células
├── prediction_result.png          # Resultado de predicción
├── venv/                          # Entorno virtual Python
├── requirements.txt               # Dependencias del proyecto
└── README.md                      # Este archivo
```

---

## Configuración del Entorno

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

## Dataset (DSB2018)

### Preparación
1. **Descargar** dataset desde Kaggle: `2018 Data Science Bowl`
2. **Descomprimir** en carpeta `data/`
3. **Verificar** estructura:

```bash
python tests/analyze_dataset.py
```

### Características del Dataset
- **2,224 muestras** - Robusto para entrenamiento
- **Dimensiones fijas**: 256×256 píxeles
- **Máscaras binarias**: Núcleos consolidados (0=fondo, 255=núcleo)
- **Desbalance de clases**: 12.2% núcleos vs 87.8% fondo
- **Variedad**: Diferentes técnicas de tinción y tipos celulares

---

## Componentes Implementados

### Etapa 1: Análisis Exploratorio
- **EDA completo** del dataset DSB2018
- **Visualizaciones** estadísticas de distribuciones
- **Identificación** de patrones y desafíos
- **Análisis** de calidad de máscaras

### Etapa 2: Ingeniería de Datos

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

### Etapa 3: Arquitectura U-Net
```python
from src.conv_blocks import BasicUNet

# Modelo U-Net desde cero con 24M parámetros
model = BasicUNet(in_channels=3, out_channels=1)
print(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
```

### Etapa 4: Entrenamiento Completo
```bash
# Entrenar el modelo (configurado para GPU)
python -m src.train

# Resultados obtenidos:
# - 91% Dice Score final
# - Convergencia en 25 épocas
# - Modelo guardado en unet_nuclei.pth
```

### Etapa 5: Conteo de Células
```python
from src.inference import NucleiPredictor

# Cargar modelo entrenado
predictor = NucleiPredictor("unet_nuclei.pth")

# Hacer predicción y conteo
result = predictor.predict_from_path("imagen.png")
count_result = predictor.count_cells(result['binary_mask'])

print(f"Células detectadas: {count_result['count']}")
# Salida: Células detectadas: 18
```

---

## Pruebas y Validación

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

## Especificaciones Técnicas

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


## Uso Rápido del Sistema Completo

### Entrenar el Modelo
```bash
# Entrenar desde cero (requiere GPU recomendado)
python -m src.train

# Resultados esperados:
# - Dice Score: ~90%
# - Modelo guardado: unet_nuclei.pth
```

### Conteo de Células en Nueva Imagen
```python
from src.inference import NucleiPredictor

# 1. Cargar modelo entrenado
predictor = NucleiPredictor("unet_nuclei.pth")

# 2. Procesar imagen y contar células
result = predictor.predict_from_path("mi_imagen.png")
count_result = predictor.count_cells(result['binary_mask'])

print(f"🔬 Células detectadas: {count_result['count']}")
print(f"📊 Área promedio: {np.mean([p.area for p in count_result['cell_properties']]):.1f} píxeles")

# 3. Visualizar resultados (se guarda automáticamente)
# Las visualizaciones se guardan en cell_counting_result.png
```

### Pipeline Completo de Datos
```python
from src.nuclei_dataset import NucleiDataset
from torch.utils.data import DataLoader

# Dataset con todas las configuraciones optimizadas
dataset = NucleiDataset(
    data_root="data", 
    normalization_method='per_channel',
    generate_weight_maps=True
)

# Listo para entrenamiento
loader = DataLoader(dataset, batch_size=4, shuffle=True)
```

---

## Información de Contacto

**Autor**: Tomás Hermosilla
**Contacto**: tomas.hermosilla.p@gmail.com 

---

## Licencia

Este proyecto abierto es de propósito educativo y de investigación.