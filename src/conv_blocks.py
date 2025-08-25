"""
Módulo de bloques convolucionales para arquitectura U-Net.

Este módulo implementa los componentes básicos de construcción para redes convolucionales:
- ConvBlock: Bloque básico Conv2d + BatchNorm2d + ReLU
- DoubleConv: Dos convoluciones consecutivas (patrón estándar U-Net)
- DownBlock: Bloque de reducción de resolución
- UpBlock: Bloque de aumento de resolución

Cada bloque está diseñado para ser modular y reutilizable.

Autor: Proyecto de Segmentación de Núcleos
Fecha: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ConvBlock(nn.Module):
    """
    Bloque convolucional básico: Conv2d → BatchNorm2d → ReLU
    
    Este es el bloque fundamental que se reutiliza en toda la arquitectura U-Net.
    Implementa el patrón estándar de capas convolucionales modernas.
    
    Componentes:
    - Conv2d: Extrae características mediante filtros aprendibles
    - BatchNorm2d: Normaliza activaciones para entrenamiento estable
    - ReLU: Introduce no-linealidad esencial para patrones complejos
    """
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 bias: bool = False):
        """
        Inicializa el bloque convolucional.
        
        Args:
            in_channels: Número de canales de entrada
            out_channels: Número de canales de salida  
            kernel_size: Tamaño del filtro convolucional (default: 3)
            stride: Paso de la convolución (default: 1, sin reducción)
            padding: Relleno para mantener dimensiones (default: 1)
            bias: Si usar bias en Conv2d (default: False, ya que BatchNorm lo compensa)
        """
        super().__init__()
        
        # Guardar parámetros para debugging
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Bloque secuencial: Conv → BatchNorm → ReLU
        self.conv_block = nn.Sequential(
            # Conv2d: El "detector de patrones"
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias  # False porque BatchNorm compensa
            ),
            
            # BatchNorm2d: El "estandarizador"
            nn.BatchNorm2d(out_channels),
            
            # ReLU: El "filtro de positividad"
            nn.ReLU(inplace=True)  # inplace=True para eficiencia de memoria
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagación hacia adelante.
        
        Args:
            x: Tensor de entrada con shape (batch_size, in_channels, height, width)
            
        Returns:
            Tensor de salida con shape (batch_size, out_channels, height, width)
            Nota: height y width se mantienen iguales si padding=1 y stride=1
        """
        return self.conv_block(x)
    
    def __repr__(self) -> str:
        """Representación legible para debugging."""
        return (f"ConvBlock(in={self.in_channels}, out={self.out_channels}, "
                f"kernel={self.kernel_size})")


class DoubleConv(nn.Module):
    """
    Bloque de doble convolución: [Conv-BN-ReLU] → [Conv-BN-ReLU]
    
    Este es el patrón estándar de U-Net original. Cada "bloque" en U-Net
    consiste en dos convoluciones consecutivas, permitiendo que cada nivel
    extraiga características más complejas progresivamente.
    
    Ventajas del patrón doble:
    - Mayor poder representacional que una sola convolución
    - Permite extraer características más complejas
    - Mantiene resolución espacial constante
    - Es el estándar probado en U-Net
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 mid_channels: Optional[int] = None,
                 kernel_size: int = 3,
                 padding: int = 1):
        """
        Inicializa el bloque de doble convolución.
        
        Args:
            in_channels: Canales de entrada
            out_channels: Canales de salida
            mid_channels: Canales intermedios (default: out_channels)
            kernel_size: Tamaño del filtro (default: 3)
            padding: Relleno (default: 1)
        """
        super().__init__()
        
        # Si no se especifica, los canales intermedios = canales de salida
        if mid_channels is None:
            mid_channels = out_channels
        
        # Guardar parámetros para debugging
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        
        # Primera convolución: in_channels → mid_channels
        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        
        # Segunda convolución: mid_channels → out_channels
        self.conv2 = ConvBlock(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagación hacia adelante: dos convoluciones consecutivas.
        
        Args:
            x: Tensor de entrada (batch_size, in_channels, height, width)
            
        Returns:
            Tensor de salida (batch_size, out_channels, height, width)
            Las dimensiones espaciales (height, width) se mantienen
        """
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
    def __repr__(self) -> str:
        """Representación legible para debugging."""
        return (f"DoubleConv(in={self.in_channels}, mid={self.mid_channels}, "
                f"out={self.out_channels})")


class DownBlock(nn.Module):
    """
    Bloque de downsampling para la parte encoder de U-Net.
    
    Combina:
    1. MaxPool2d: Reduce resolución espacial (256→128→64→32...)
    2. DoubleConv: Extrae características en la nueva resolución
    
    Este patrón permite que la red capture características a múltiples escalas:
    - Escalas altas: Detalles finos (bordes precisos)
    - Escalas bajas: Patrones globales (forma general de núcleos)
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 pool_size: int = 2):
        """
        Inicializa el bloque de downsampling.
        
        Args:
            in_channels: Canales de entrada
            out_channels: Canales de salida
            pool_size: Factor de reducción (default: 2, mitad del tamaño)
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool_size = pool_size
        
        # MaxPool2d: Reduce resolución espacial
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
        
        # DoubleConv: Procesa características en nueva resolución
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagación hacia adelante: pooling → double convolution.
        
        Args:
            x: Tensor de entrada (batch, in_channels, height, width)
            
        Returns:
            Tensor de salida (batch, out_channels, height//pool_size, width//pool_size)
        """
        x = self.pool(x)
        x = self.conv(x)
        return x
    
    def __repr__(self) -> str:
        return (f"DownBlock(in={self.in_channels}, out={self.out_channels}, "
                f"pool={self.pool_size})")


class UpBlock(nn.Module):
    """
    Bloque de upsampling para la parte decoder de U-Net.
    
    Combina:
    1. ConvTranspose2d: Aumenta resolución espacial (32→64→128→256...)
    2. Concatenación: Fusiona con skip connection del encoder
    3. DoubleConv: Procesa características fusionadas
    
    Las skip connections son cruciales en segmentación porque:
    - Preservan detalles espaciales finos
    - Permiten localización precisa de píxeles
    - Combinan información de múltiples escalas
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 bilinear: bool = False):
        """
        Inicializa el bloque de upsampling.
        
        Args:
            in_channels: Canales de entrada
            out_channels: Canales de salida
            bilinear: Si usar interpolación bilineal en lugar de ConvTranspose2d
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # Upsampling: Aumentar resolución espacial
        if bilinear:
            # Opción 1: Interpolación bilinear + Conv1x1
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # Para bilinear: no cambian canales, así que concat será in_channels + skip_channels
            # Asumo que skip tiene out_channels, entonces total = in_channels + out_channels
            self.conv = DoubleConv(in_channels + out_channels, out_channels)
        else:
            # Opción 2: ConvTranspose2d (transposed convolution)
            self.up = nn.ConvTranspose2d(
                in_channels, 
                in_channels // 2, 
                kernel_size=2, 
                stride=2
            )
            # Para ConvTranspose2d: reduce a in_channels//2, skip tiene out_channels
            # Total después de concat = in_channels//2 + out_channels
            self.conv = DoubleConv(in_channels // 2 + out_channels, out_channels)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Propagación hacia adelante con skip connection.
        
        Args:
            x1: Tensor del decoder (resolución baja)
            x2: Tensor del encoder - skip connection (resolución alta)
            
        Returns:
            Tensor fusionado y procesado
        """
        # 1. Upsampling del tensor de resolución baja
        x1 = self.up(x1)
        
        # Validaciones de canales para errores más descriptivos (ruta ConvTranspose2d)
        # En el caso no bilinear, tras el upsampling:
        #  - x1 debe tener self.in_channels // 2 canales después del ConvTranspose2d
        #  - x2 (skip) puede tener cualquier número de canales 
        #  - Al concatenar: x1_channels + x2_channels debe igualar el input esperado por self.conv
        if not self.bilinear:
            expected_x1_channels = self.in_channels // 2
            c1 = x1.size(1)
            c2 = x2.size(1)
            total_channels = c1 + c2
            
            # Solo validar que x1 tenga los canales correctos tras upsampling
            if c1 != expected_x1_channels:
                raise RuntimeError(
                    (
                        f"UpBlock upsampling error. After ConvTranspose2d, x1 should have "
                        f"{expected_x1_channels} channels, but got {c1}.\n"
                        f"Input to UpBlock: {self.in_channels} channels\n"
                        f"Skip connection: {c2} channels\n"
                        f"Total after concat: {total_channels} channels"
                    )
                )
        
        # 2. Ajustar dimensiones si hay diferencias menores
        # (puede ocurrir por diferencias de padding en el encoder)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                           diff_y // 2, diff_y - diff_y // 2])
        
        # 3. Concatenar a lo largo del canal
        x = torch.cat([x2, x1], dim=1)
        
        # 4. Procesar características fusionadas
        return self.conv(x)
    
    def __repr__(self) -> str:
        return (f"UpBlock(in={self.in_channels}, out={self.out_channels}, "
                f"bilinear={self.bilinear})")


class BasicUNet(nn.Module):
    """
    U-Net básica para segmentación de núcleos celulares.
    
    Arquitectura simplificada con 3 niveles:
    - Encoder: 3 DownBlocks (256→128→64→32)
    - Bottleneck: DoubleConv central
    - Decoder: 3 UpBlocks (32→64→128→256) con skip connections
    - Output: Convolución final para segmentación binaria
    
    Esta implementación sigue el patrón del paper original U-Net
    adaptado para nuestro problema de segmentación de núcleos.
    """
    
    def __init__(self, 
                 in_channels: int = 3, 
                 out_channels: int = 1,
                 bilinear: bool = False):
        """
        Inicializa la U-Net básica.
        
        Args:
            in_channels: Canales de entrada (3 para RGB)
            out_channels: Canales de salida (1 para segmentación binaria)
            bilinear: Si usar interpolación bilinear en lugar de ConvTranspose2d
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # Encoder (Contraction Path) - Configuración estándar U-Net
        # Nivel 0: Input conv (sin pooling)  
        self.inc = DoubleConv(in_channels, 64)
        
        # Nivel 1: 256x256 → 128x128
        self.down1 = DownBlock(64, 128)
        
        # Nivel 2: 128x128 → 64x64  
        self.down2 = DownBlock(128, 256)
        
        # Nivel 3: 64x64 → 32x32
        self.down3 = DownBlock(256, 512)
        
        # Bottleneck: 32x32 (contexto global máximo)
        factor = 2 if bilinear else 1
        self.bottleneck = DoubleConv(512, 1024 // factor)
        
        # Decoder (Expansion Path) 
        # up1: bottleneck(1024) → 512, skip de down2(256)
        # UpBlock espera: in_channels=1024, skip=512 (1024//2)
        # Pero down2 solo da 256, entonces ajusto para que funcione
        self.up1 = UpBlock(1024 // factor, 256, bilinear)
        
        # up2: up1(256) → 128, skip de down1(128)  
        self.up2 = UpBlock(256, 128, bilinear)
        
        # up3: up2(128) → 64, skip de inc(64)
        self.up3 = UpBlock(128, 64, bilinear)
        
        # Output: Convolución final para clasificación por píxel
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de la U-Net.
        
        Args:
            x: Tensor de entrada (batch_size, in_channels, height, width)
            
        Returns:
            Tensor de salida (batch_size, out_channels, height, width)
            Para segmentación binaria: valores entre 0 y 1 (aplicar sigmoid)
        """
        # Encoder: Extracción de características con skip connections
        x1 = self.inc(x)          # 64 channels, 256x256
        x2 = self.down1(x1)       # 128 channels, 128x128
        x3 = self.down2(x2)       # 256 channels, 64x64
        x4 = self.down3(x3)       # 512 channels, 32x32
        
        # Bottleneck: Máximo contexto global
        x5 = self.bottleneck(x4)  # 1024 channels, 32x32
        
        # Decoder: Reconstrucción con skip connections
        x = self.up1(x5, x3)      # 256 channels, 64x64 (skip: x3)
        x = self.up2(x, x2)       # 128 channels, 128x128 (skip: x2)
        x = self.up3(x, x1)       # 64 channels, 256x256 (skip: x1)
        
        # Output: Clasificación por píxel
        logits = self.outc(x)     # out_channels, 256x256
        
        return logits
    
    def get_architecture_summary(self) -> dict:
        """
        Retorna resumen detallado de la arquitectura.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': 'BasicUNet',
            'input_shape': f"({self.in_channels}, 256, 256)",
            'output_shape': f"({self.out_channels}, 256, 256)",
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_size_mb': total_params * 4 / (1024 * 1024),  # float32 = 4 bytes
            'bilinear_upsampling': self.bilinear,
            'encoder_channels': [64, 128, 256, 512],
            'bottleneck_channels': 1024 // (2 if self.bilinear else 1),
            'decoder_channels': [256, 128, 64],
            'skip_connections': 3
        }
    
    def __repr__(self) -> str:
        summary = self.get_architecture_summary()
        return (f"BasicUNet(\n"
                f"  input_channels={self.in_channels},\n"
                f"  output_channels={self.out_channels},\n" 
                f"  parameters={summary['total_parameters']:,},\n"
                f"  size_mb={summary['parameter_size_mb']:.1f}MB\n"
                f")")


def test_basic_unet():



    
    """
    Función para probar la U-Net básica con datos reales.
    """
    print("🏗️ Probando BasicUNet...")
    
    # Crear modelo
    model = BasicUNet(in_channels=3, out_channels=1)
    model.eval()  # Modo evaluación para testing
    
    # Mostrar resumen de arquitectura
    summary = model.get_architecture_summary()
    print(f"\n📋 Resumen de arquitectura:")
    for key, value in summary.items():
        if isinstance(value, list):
            value = f"[{', '.join(map(str, value))}]"
        print(f"   {key:20s}: {value}")
    
    # Test con tensor sintético
    print(f"\n🧪 Test con datos sintéticos:")
    batch_size = 2
    test_input = torch.randn(batch_size, 3, 256, 256)
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"   Input shape:  {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Verificar dimensiones esperadas
    expected_shape = (batch_size, 1, 256, 256)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    # Test con imagen real del dataset (si existe)
    try:
        from src.nuclei_dataset import NucleiDataset
        print(f"\n🔬 Test con imagen real del dataset:")
        
        dataset = NucleiDataset('/home/tomy/Documents/ML-AI-Projects/data')
        if len(dataset) > 0:
            # El dataset devuelve un diccionario, no una tupla
            sample = dataset[0]
            print(f"   Dataset sample keys: {list(sample.keys())}")

            # Extraer la imagen del diccionario
            real_image = sample.get('image')

            # Verificar que la imagen se haya cargado correctamente
            if real_image is None:
                print("   ⚠️ No se encontró la clave 'image' en la muestra del dataset.")
                return
            
            print(f"   Image type: {type(real_image)}, shape: {getattr(real_image, 'shape', 'N/A')}")
            
            # Verificar que sea un tensor antes de continuar
            if not isinstance(real_image, torch.Tensor):
                print(f"   ⚠️ 'real_image' no es un tensor de PyTorch, es de tipo {type(real_image)}.")
                return
            
            # Añadir dimensión de batch
            real_input = real_image.unsqueeze(0)  # (1, 3, 256, 256)
            
            with torch.no_grad():
                real_output = model(real_input)
            
            print(f"   Real input shape:  {real_input.shape}")
            print(f"   Real output shape: {real_output.shape}")
            print(f"   Real output range: [{real_output.min():.3f}, {real_output.max():.3f}]")
            
            # Aplicar sigmoid para probabilidades
            probs = torch.sigmoid(real_output)
            print(f"   Probabilities:     [{probs.min():.3f}, {probs.max():.3f}]")
            
            return {
                'model': model,
                'synthetic_output': output,
                'real_input': real_input,
                'real_output': real_output,
                'probabilities': probs
            }
        else:
            print("   ⚠️ Dataset vacío, omitiendo test con datos reales")
            
    except ImportError:
        print("   ⚠️ NucleiDataset no disponible, omitiendo test con datos reales")
    
    print(f"\n✅ BasicUNet: Todas las pruebas exitosas!")
    
    return {
        'model': model,
        'synthetic_output': output,
        'summary': summary
    }


def visualize_unet_architecture():
    """
    Crea visualización detallada de la arquitectura U-Net.
    """
    print("📐 Visualizando arquitectura U-Net...")
    
    model = BasicUNet()
    
    # Simular forward pass para capturar shapes
    x = torch.randn(1, 3, 256, 256)
    
    shapes = []
    
    # Hook para capturar shapes intermedios
    def capture_shape(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                shapes.append((name, output.shape))
        return hook
    
    # Registrar hooks
    model.inc.register_forward_hook(capture_shape("Initial Conv"))
    model.down1.register_forward_hook(capture_shape("Down 1"))
    model.down2.register_forward_hook(capture_shape("Down 2")) 
    model.down3.register_forward_hook(capture_shape("Down 3"))
    model.bottleneck.register_forward_hook(capture_shape("Bottleneck"))
    model.up1.register_forward_hook(capture_shape("Up 1"))
    model.up2.register_forward_hook(capture_shape("Up 2"))
    model.up3.register_forward_hook(capture_shape("Up 3"))
    model.outc.register_forward_hook(capture_shape("Output Conv"))
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"\n🔄 Flujo de datos a través de la red:")
    print(f"   {'Layer':<15} {'Shape':<20} {'Channels':<10} {'Resolution':<12}")
    print(f"   {'-'*60}")
    
    shapes.insert(0, ("Input", x.shape))
    shapes.append(("Final Output", output.shape))
    
    for name, shape in shapes:
        channels = shape[1] if len(shape) > 1 else "N/A"
        resolution = f"{shape[2]}x{shape[3]}" if len(shape) > 3 else "N/A"
        print(f"   {name:<15} {str(shape):<20} {channels:<10} {resolution:<12}")
    
    # Mostrar conexiones skip
    print(f"\n🔗 Skip Connections:")
    print(f"   • Initial Conv (64ch, 256x256) → Up 3")
    print(f"   • Down 1 (128ch, 128x128) → Up 2") 
    print(f"   • Down 2 (256ch, 64x64) → Up 1")
    
    # Calcular receptive field aproximado
    rf = 1
    for i in range(3):  # 3 niveles de pooling
        rf = rf * 2 + 2  # Cada pooling + conv aumenta RF
    
    print(f"\n👁️ Receptive Field aproximado: {rf}x{rf} píxeles")
    print(f"   (cada píxel de salida 've' una región de {rf}x{rf} en la entrada)")
    
    return {
        'model': model,
        'layer_shapes': shapes,
        'receptive_field': rf
    }


def test_downsampling_operations():
    """
    Función para probar operaciones de downsampling paso a paso.
    Verifica la reducción de resolución: 256→128→64→32
    """
    print("🔽 Probando operaciones de downsampling...")
    
    # Tensor de prueba inicial
    x = torch.randn(2, 64, 256, 256)
    print(f"Input inicial: {x.shape}")
    
    # Serie de MaxPool2d para reducir resolución
    pool_layers = []
    current_size = 256
    
    for i in range(3):  # 3 reducciones: 256→128→64→32
        pool = nn.MaxPool2d(kernel_size=2, stride=2)
        pool_layers.append(pool)
        
        x = pool(x)
        current_size = current_size // 2
        print(f"   Downsampling {i+1}: {x.shape} (resolución: {current_size}x{current_size})")
    
    # Verificar que llegamos a 32x32
    assert x.shape[-1] == 32, f"Se esperaba resolución final 32x32, pero se obtuvo {x.shape[-2]}x{x.shape[-1]}"
    
    print("✅ Downsampling: 256→128→64→32 completado exitosamente!")
    return pool_layers, x


def test_upsampling_operations():
    """
    Función para probar operaciones de upsampling paso a paso.
    Verifica el aumento de resolución: 32→64→128→256
    """
    print("🔼 Probando operaciones de upsampling...")
    
    # Tensor de prueba inicial (desde bottleneck)
    x = torch.randn(2, 512, 32, 32)
    print(f"Input inicial (bottleneck): {x.shape}")
    
    # Serie de ConvTranspose2d para aumentar resolución
    upsample_layers = []
    channel_progression = [512, 256, 128, 64]  # Reducción de canales típica
    
    for i in range(3):  # 3 aumentos: 32→64→128→256
        in_ch = channel_progression[i]
        out_ch = channel_progression[i + 1]
        
        # ConvTranspose2d para upsampling
        upsample = nn.ConvTranspose2d(
            in_channels=in_ch,
            out_channels=out_ch, 
            kernel_size=2,
            stride=2
        )
        upsample_layers.append(upsample)
        
        x = upsample(x)
        current_size = x.shape[-1]
        print(f"   Upsampling {i+1}: {x.shape} (resolución: {current_size}x{current_size})")
    
    # Verificar que llegamos a 256x256
    assert x.shape[-1] == 256, f"Se esperaba resolución final 256x256, pero se obtuvo {x.shape[-2]}x{x.shape[-1]}"
    
    print("✅ Upsampling: 32→64→128→256 completado exitosamente!")
    return upsample_layers, x


def test_information_preservation():
    """
    Función para analizar preservación vs pérdida de información en down/up sampling.
    """
    print("📊 Analizando preservación de información...")
    
    # Crear imagen sintética con patrón conocido
    batch_size = 1
    x_original = torch.zeros(batch_size, 3, 256, 256)
    
    # Crear patrón de círculos (simula núcleos)
    for i in range(0, 256, 64):
        for j in range(0, 256, 64):
            center_i, center_j = i + 32, j + 32
            for di in range(-16, 17):
                for dj in range(-16, 17):
                    if di*di + dj*dj <= 256:  # Círculo de radio 16
                        if 0 <= center_i + di < 256 and 0 <= center_j + dj < 256:
                            x_original[0, :, center_i + di, center_j + dj] = 1.0
    
    print(f"Imagen original: {x_original.shape}, valores únicos: {torch.unique(x_original)}")
    
    # Downsampling progresivo
    x_down = x_original
    sizes = [256]
    for level in range(3):
        pool = nn.MaxPool2d(kernel_size=2, stride=2)
        x_down = pool(x_down)
        sizes.append(x_down.shape[-1])
        unique_vals = len(torch.unique(x_down))
        print(f"   Nivel {level+1} ({x_down.shape[-1]}x{x_down.shape[-1]}): {unique_vals} valores únicos")
    
    # Upsampling con interpolación bilinear (preserva mejor info que ConvTranspose sin entrenamiento)
    x_up = x_down
    for level in range(3):
        target_size = sizes[-(level+2)]  # Tamaños en orden inverso
        x_up = F.interpolate(x_up, size=(target_size, target_size), mode='bilinear', align_corners=True)
        unique_vals = len(torch.unique(x_up))
        print(f"   Upsampling a {target_size}x{target_size}: {unique_vals} valores únicos")
    
    # Calcular pérdida de información
    mse_loss = F.mse_loss(x_up, x_original)
    print(f"📉 MSE Loss (original vs reconstruida): {mse_loss:.6f}")
    
    # Análisis de preservación de bordes
    def edge_count(tensor):
        """Cuenta transiciones de 0 a 1 (aproximación de bordes)"""
        diff_h = torch.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :])
        diff_w = torch.abs(tensor[:, :, :, 1:] - tensor[:, :, :, :-1])
        return (diff_h > 0.1).sum() + (diff_w > 0.1).sum()
    
    edges_original = edge_count(x_original)
    edges_reconstructed = edge_count((x_up > 0.5).float())  # Binarizar reconstrucción
    edge_preservation = edges_reconstructed.float() / edges_original.float() * 100
    
    print(f"🔍 Preservación de bordes: {edge_preservation:.1f}% ({edges_reconstructed}/{edges_original})")
    
    return {
        'original': x_original,
        'reconstructed': x_up,
        'mse_loss': mse_loss.item(),
        'edge_preservation_percent': edge_preservation.item()
    }


def analyze_computational_costs():
    """
    Analiza los costos computacionales de diferentes operaciones.
    """
    print("💻 Analizando costos computacionales...")
    
    import time
    import psutil
    import os
    
    # Tensor de prueba
    x = torch.randn(4, 256, 128, 128)  # Batch típico
    
    operations = {
        'MaxPool2d': nn.MaxPool2d(kernel_size=2, stride=2),
        'ConvTranspose2d': nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
        'Bilinear Upsample': lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True),
        'Nearest Upsample': lambda x: F.interpolate(x, scale_factor=2, mode='nearest')
    }
    
    results = {}
    
    for name, op in operations.items():
        # Medir tiempo
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        for _ in range(100):  # Promedio de 100 ejecuciones
            with torch.no_grad():
                if callable(op):
                    _ = op(x)
                else:
                    _ = op(x)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / 100 * 1000  # ms
        
        # Calcular parámetros (solo para capas con parámetros)
        params = 0
        if hasattr(op, 'parameters'):
            params = sum(p.numel() for p in op.parameters())
        
        results[name] = {
            'avg_time_ms': avg_time,
            'parameters': params
        }
        
        print(f"   {name:20s}: {avg_time:.3f}ms, {params:,} parámetros")
    
    return results


def test_conv_blocks():
    """
    Función de prueba para validar todos los bloques convolucionales.
    """
    print("🧪 Probando bloques convolucionales...")
    
    # Crear tensor de prueba (simula batch de imágenes de núcleos)
    batch_size = 2
    test_input = torch.randn(batch_size, 3, 256, 256)
    print(f"Input de prueba: {test_input.shape}")
    
    # 1. Probar ConvBlock básico
    print("\n1. Probando ConvBlock:")
    conv_block = ConvBlock(in_channels=3, out_channels=64)
    output1 = conv_block(test_input)
    print(f"   {conv_block}")
    print(f"   Input: {test_input.shape} → Output: {output1.shape}")
    
    # 2. Probar DoubleConv
    print("\n2. Probando DoubleConv:")
    double_conv = DoubleConv(in_channels=3, out_channels=64)
    output2 = double_conv(test_input)
    print(f"   {double_conv}")
    print(f"   Input: {test_input.shape} → Output: {output2.shape}")
    
    # 3. Probar DownBlock
    print("\n3. Probando DownBlock:")
    down_block = DownBlock(in_channels=3, out_channels=64)
    output3 = down_block(test_input)
    print(f"   {down_block}")
    print(f"   Input: {test_input.shape} → Output: {output3.shape}")
    
    # 4. Probar UpBlock
    print("\n4. Probando UpBlock:")
    # Simular tensores para skip connection
    x1 = torch.randn(batch_size, 128, 64, 64)  # Del decoder
    x2 = torch.randn(batch_size, 64, 128, 128)  # Skip connection del encoder
    
    up_block = UpBlock(in_channels=128, out_channels=64)
    output4 = up_block(x1, x2)
    print(f"   {up_block}")
    print(f"   Decoder: {x1.shape} + Skip: {x2.shape} → Output: {output4.shape}")
    
    print("\n✅ Todas las pruebas completadas exitosamente!")
    
    return {
        'conv_block': conv_block,
        'double_conv': double_conv, 
        'down_block': down_block,
        'up_block': up_block
    }


def test_all_sampling_operations():
    """
    Función principal que ejecuta todas las pruebas de sampling.
    """
    print("🚀 Ejecutando pruebas completas de operaciones de sampling...\n")
    
    # 1. Probar downsampling
    pool_layers, down_result = test_downsampling_operations()
    print()
    
    # 2. Probar upsampling  
    upsample_layers, up_result = test_upsampling_operations()
    print()
    
    # 3. Analizar preservación de información
    info_analysis = test_information_preservation()
    print()
    
    # 4. Analizar costos computacionales
    cost_analysis = analyze_computational_costs()
    print()
    
    print("🎯 Resumen de hallazgos:")
    print(f"   • Downsampling exitoso: 256→128→64→32")
    print(f"   • Upsampling exitoso: 32→64→128→256") 
    print(f"   • MSE Loss reconstrucción: {info_analysis['mse_loss']:.6f}")
    print(f"   • Preservación de bordes: {info_analysis['edge_preservation_percent']:.1f}%")
    print(f"   • MaxPool2d más rápido que ConvTranspose2d")
    print(f"   • Interpolación bilineal: balance velocidad/calidad")
    
    return {
        'downsampling': {'layers': pool_layers, 'result': down_result},
        'upsampling': {'layers': upsample_layers, 'result': up_result},
        'information_analysis': info_analysis,
        'computational_costs': cost_analysis
    }


if __name__ == "__main__":
    # Ejecutar pruebas cuando se corre directamente
    test_conv_blocks()
