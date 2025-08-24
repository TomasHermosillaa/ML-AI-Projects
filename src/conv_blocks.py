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
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # Opción 2: ConvTranspose2d (transposed convolution)
            self.up = nn.ConvTranspose2d(
                in_channels, 
                in_channels // 2, 
                kernel_size=2, 
                stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)
    
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


if __name__ == "__main__":
    # Ejecutar pruebas cuando se corre directamente
    test_conv_blocks()
