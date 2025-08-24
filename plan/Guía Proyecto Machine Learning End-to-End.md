

# **Proyecto de IA de Calidad para Portafolio: Segmentación de Instancias de Núcleos de Extremo a Extremo**

## **Parte I: El Marco Fundacional \- Ingeniería de Datos para Imágenes de Microscopía**

La base de cualquier proyecto de aprendizaje profundo exitoso reside en una ingeniería de datos meticulosa y bien razonada. En la segmentación de imágenes biomédicas, este principio es aún más crítico. Esta sección detalla los pasos para establecer el entorno, adquirir y procesar el conjunto de datos DSB2018, y construir un pipeline de datos robusto que aborde los desafíos únicos de la segmentación de instancias.

### **Configuración del Entorno**

Un entorno de desarrollo limpio, aislado y reproducible es esencial. El uso de entornos virtuales es una práctica estándar que evita conflictos entre las dependencias de diferentes proyectos. La estrategia recomendada es utilizar una herramienta como venv (incluida en Python) o conda para crear un entorno aislado y luego definir todas las dependencias en un archivo requirements.txt. Este enfoque es agnóstico a la herramienta y es el método estándar para compartir proyectos de Python de manera reproducible. Asegúrate de instalar PyTorch primero, seleccionando la versión compatible con tu configuración de CUDA desde su sitio web oficial, y luego instala el resto de las bibliotecas (como OpenCV, scikit-image, etc.) desde tu archivo de requisitos.

### **Adquisición y Exploración de Datos**

El proyecto utilizará el conjunto de datos del **2018 Data Science Bowl (DSB2018)**.2 Este conjunto de datos es ideal porque fue diseñado para desafiar a los algoritmos a generalizar a través de una amplia variedad de tipos de células, modalidades de imagen y condiciones experimentales.20

Tu primer paso es descargar y explorar la estructura de los datos. Notarás que para cada imagen de entrenamiento, no hay una sola máscara, sino una carpeta que contiene múltiples archivos de máscara, cada uno para un núcleo individual.21 Este es un detalle crucial. Antes de cualquier procesamiento, realiza un Análisis Exploratorio de Datos (EDA) para visualizar las imágenes y sus máscaras, y para entender la distribución de tamaños de imagen, que son variables y requerirán estandarización.16

### **El Desafío Central: Consolidación de Máscaras para la Conciencia de Instancias**

Este es el paso de ingeniería de datos más crítico. Si simplemente fusionas todas las máscaras individuales en una sola máscara binaria, el modelo aprenderá a predecir una única mancha para los núcleos que se tocan, fallando en la tarea de segmentación de instancias.5 La solución, inspirada en el paper original de U-Net, es enseñar al modelo a reconocer los límites entre núcleos adyacentes.

Para lograr esto, tu pipeline de datos debe generar dos salidas para cada imagen de entrada:

1. **Máscara Semántica Combinada:** Una única máscara binaria donde un píxel es 1 si pertenece a *cualquier* núcleo.  
2. **Mapa de Pesos de Bordes:** Un mapa donde los píxeles en los bordes entre núcleos que se tocan tienen un peso significativamente mayor. Esto penalizará más al modelo si se equivoca en estas áreas críticas, forzándolo a aprender a separar las instancias.

La implementación de esta lógica requerirá cargar todas las máscaras individuales, combinarlas y luego aplicar técnicas de morfología de imágenes (como erosión y dilatación) para identificar y ponderar los píxeles de borde entre instancias cercanas.16

### **Pre-procesamiento Avanzado**

#### **Normalización de Tinción**

Las imágenes de microscopía presentan una gran variabilidad de color debido a diferencias en los protocolos de tinción. Para evitar que el modelo interprete estas variaciones como características biológicas, debes aplicar **normalización de tinción**. Esta técnica estandariza el espacio de color de las imágenes. Bibliotecas como torchstain ofrecen implementaciones de algoritmos de normalización que puedes integrar en tu pipeline.36

#### **Redimensionamiento y Relleno de Imágenes**

Los modelos convolucionales requieren entradas de tamaño fijo. Debes redimensionar todas las imágenes a una dimensión estándar (por ejemplo, 256x256 es un buen punto de partida para tu hardware).16 Es importante usar relleno (padding) para mantener la relación de aspecto original de las imágenes antes de redimensionarlas para evitar distorsiones.

### **Dataset y DataLoader Personalizados de PyTorch**

Para gestionar eficientemente la carga y el pre-procesamiento, debes encapsular toda esta lógica en una clase Dataset personalizada de PyTorch. Esta es una práctica estándar que carga y procesa cada muestra sobre la marcha, en lugar de cargar todo el conjunto de datos en la RAM.37 El método \_\_getitem\_\_ de esta clase será el núcleo de tu pipeline, donde se aplicarán secuencialmente la carga de imágenes, la normalización de tinción, la consolidación de máscaras, el redimensionamiento y los aumentos de datos. Finalmente, el DataLoader de PyTorch se encargará de crear lotes de datos, aleatorizarlos y cargarlos en paralelo para alimentar eficientemente la GPU.

## 

## 

## 

## 

## **Parte II: El Plan Arquitectónico \- Implementando el Modelo U-Net**

Una vez que el pipeline de datos está establecido, el siguiente paso es construir la arquitectura de la red neuronal que aprenderá a realizar la segmentación. Para la segmentación de imágenes biomédicas, la arquitectura U-Net no es solo una opción, es el estándar de oro y un punto de partida fundamental debido a su diseño elegante y altamente efectivo.13

### **Inmersión Profunda en la Arquitectura U-Net**

La U-Net, propuesta por Ronneberger et al. en 2015, fue diseñada específicamente para la segmentación de imágenes biomédicas.13 Su arquitectura en forma de "U" se divide en tres componentes principales 14:

1. **Codificador (Ruta Contractiva):** La mitad izquierda de la "U". Sigue la estructura de una CNN típica, utilizando bloques de convoluciones y max-pooling para extraer características cada vez más complejas y reducir progresivamente las dimensiones espaciales. Esto permite a la red capturar información contextual de un campo receptivo más amplio.  
2. **Cuello de Botella (Bottleneck):** El punto más bajo de la "U", que conecta el codificador con el decodificador. Procesa los mapas de características de más bajo nivel y más ricos en semántica.  
3. **Decodificador (Ruta Expansiva):** La mitad derecha de la "U". Su objetivo es reconstruir una máscara de segmentación de alta resolución. Para ello, utiliza convoluciones transpuestas para aumentar las dimensiones espaciales.

#### **Conexiones de Salto: La Clave de la Precisión**

La característica más innovadora de la U-Net son las **conexiones de salto (skip connections)**.13 Estas conexiones concatenan directamente los mapas de características del codificador con los del decodificador en el nivel de resolución correspondiente. Esto es crucial porque el codificador pierde información espacial precisa a medida que profundiza. Las conexiones de salto reintroducen esta información de localización de alta resolución en el decodificador, permitiéndole reconstruir una segmentación que es a la vez semánticamente correcta y espacialmente precisa.

### **Implementación desde Cero**

Construir la U-Net desde cero es un ejercicio excelente para solidificar la comprensión de la arquitectura. Tu implementación debe ser modular. Comienza creando un bloque de "doble convolución" (dos convoluciones 3x3 con activación ReLU y normalización por lotes) que se repetirá a lo largo de la red. Luego, ensambla estos bloques en las rutas del codificador y del decodificador, implementando las operaciones de max-pooling y convolución transpuesta, y lo más importante, las conexiones de salto que concatenan las características entre ambas rutas. Existen muchas implementaciones de referencia de alta calidad que puedes estudiar para guiar tu propia construcción.38

### **Opción Avanzada: Attention U-Net**

Para elevar aún más el nivel del proyecto, considera implementar la **Attention U-Net**.43 Esta variante mejora la U-Net estándar al incorporar "Puertas de Atención" (Attention Gates) en las conexiones de salto. La idea es que la red aprenda a suprimir activaciones en regiones irrelevantes mientras enfatiza las características salientes. En esencia, la red aprende "dónde mirar", utilizando la información de las capas más profundas para guiar la transmisión de características desde el codificador.43 La implementación de esta variante demuestra la capacidad de comprender e implementar conceptos de investigación más allá de las arquitecturas canónicas.

## 

## 

## 

## 

## 

## 

## 

## 

## 

## **Parte III: El Régimen de Entrenamiento \- Optimización y Evaluación**

Con la arquitectura del modelo definida y el pipeline de datos listo, la siguiente fase es el entrenamiento del modelo. Este proceso es un arte y una ciencia, que requiere un diseño cuidadoso del bucle de entrenamiento, una selección informada de la función de pérdida y las métricas de evaluación, y la aplicación de técnicas de optimización cruciales para trabajar dentro de las restricciones de hardware.

### **Diseño del Bucle de Entrenamiento**

Un bucle de entrenamiento bien estructurado es la columna vertebral de cualquier proyecto de PyTorch. Debes tener una clara separación entre la lógica de entrenamiento y la de validación. Para cada época, el bucle de entrenamiento debe iterar sobre los lotes de datos, moverlos a la GPU, realizar los pases hacia adelante y hacia atrás para calcular y propagar los gradientes, y finalmente actualizar los pesos del modelo. El bucle de validación es similar, pero se ejecuta sin el cálculo de gradientes (torch.no\_grad()) para ahorrar memoria y tiempo. Es fundamental guardar un "checkpoint" del modelo solo cuando el rendimiento en el conjunto de validación mejore.

### **Funciones de Pérdida para Segmentación Desbalanceada**

La elección de la función de pérdida es fundamental. En la segmentación de núcleos, el fondo suele dominar a los núcleos, creando un desequilibrio de clases. Una Entropía Cruzada Binaria (BCE) estándar puede ser subóptima. Por lo tanto, debes explorar funciones de pérdida más robustas:

* **Pérdida Dice (Dice Loss):** Derivada directamente de la métrica de superposición del Coeficiente de Dice, es inherentemente robusta al desequilibrio de clases y funciona muy bien para la segmentación.17  
* **Pérdida Combinada Dice-BCE (DiceBCELoss):** A menudo, la mejor opción en la práctica. Combina la estabilidad de la BCE con la robustez de la pérdida Dice.17 Tu implementación debe calcular ambas pérdidas y sumarlas. Además, la parte de la BCE debe incorporar los mapas de peso que creaste en la Parte I para penalizar más los errores en los bordes de los núcleos.

### **Métrica de Evaluación: Intersección sobre Unión (IoU)**

Mientras que la función de pérdida guía el entrenamiento, necesitas una métrica objetiva para medir el rendimiento. Para la segmentación, la métrica estándar es la **Intersección sobre Unión (IoU)**, también conocida como Índice de Jaccard.46 Se calcula como el área de la intersección entre la máscara predicha y la verdadera, dividida por el área de su unión. Un IoU de 1 indica una segmentación perfecta. Debes calcular esta métrica en el conjunto de validación después de cada época para monitorear el progreso del modelo.48

### **Crucial para 8GB VRAM: Entrenamiento con Precisión Mixta Automática (AMP)**

Este paso es **esencial** para entrenar con éxito en una GPU con 8 GB de VRAM. El entrenamiento con precisión mixta automática (AMP) utiliza una combinación de tipos de datos de 32 y 16 bits para reducir el uso de memoria y acelerar el entrenamiento en GPUs modernas.50 PyTorch simplifica enormemente su uso a través de

torch.cuda.amp.autocast y torch.cuda.amp.GradScaler.52 Debes envolver tu pase hacia adelante en el contexto

autocast y usar el GradScaler para escalar la pérdida antes de la retropropagación y desescalar los gradientes antes de la actualización del optimizador. La implementación correcta de AMP puede reducir el uso de VRAM en casi un 50%, lo que es un diferenciador clave para completar este proyecto en tu hardware.

### **Tabla 2: Hiperparámetros de Entrenamiento de U-Net**

La siguiente tabla proporciona un conjunto de hiperparámetros de referencia sólidos y reproducibles para comenzar los experimentos de entrenamiento.

| Hiperparámetro | Valor Sugerido | Justificación |
| :---- | :---- | :---- |
| **Arquitectura del Modelo** | U-Net (o Attention U-Net) | Estándar de la industria para segmentación biomédica; la variante de atención es una mejora avanzada. |
| **Tamaño de Imagen** | 256x256 píxeles | Buen equilibrio entre detalle y consumo de VRAM para una GPU de 8 GB. 16 |
| **Tamaño del Lote (Batch Size)** | 4 \- 8 | Máximo posible sin exceder la VRAM de 8 GB al usar AMP. Un tamaño de lote más grande generalmente conduce a una mejor estimación del gradiente. |
| **Optimizador** | AdamW | Un optimizador robusto que a menudo converge más rápido y mejor que el SGD estándar para tareas de visión. |
| **Tasa de Aprendizaje (LR)** | 1×10−4 | Un punto de partida común y efectivo para AdamW. Se puede utilizar un programador de LR (e.g., ReduceLROnPlateau). |
| **Función de Pérdida** | DiceBCELoss | Combina la estabilidad de BCE con la robustez al desequilibrio de clases de Dice Loss. 17 |
| **Número de Épocas** | 50 \- 100 | Suficiente para la convergencia, utilizando early stopping basado en la métrica de validación (IoU) para evitar el sobreajuste. |
| **Precisión Mixta (AMP)** | Habilitada | **Crítico.** Necesario para ajustar el modelo en 8 GB de VRAM y acelerar el entrenamiento. 50 |

## **Parte IV: De Píxeles a Objetos \- Post-Procesamiento para la Segmentación de Instancias**

El entrenamiento del modelo U-Net producirá un mapa de probabilidad, que es una predicción a nivel de píxel (segmentación semántica). El paso final y crucial es convertir este mapa en un conjunto de máscaras de instancia, donde cada núcleo detectado tiene una etiqueta de identificación única. Aquí es donde un algoritmo clásico de visión por computadora, el **algoritmo de Watershed**, se convierte en un complemento poderoso.

### **El Puente de lo Semántico a la Instancia**

La integración de un algoritmo clásico como Watershed con la salida de un modelo de aprendizaje profundo es una técnica poderosa y común en aplicaciones del mundo real. Demuestra la capacidad de construir pipelines híbridos, utilizando las fortalezas de cada enfoque. Incluir este paso es un indicador de madurez y pragmatismo en la resolución de problemas.

### **Inmersión Profunda en el Algoritmo de Watershed**

#### **Teoría**

El algoritmo de Watershed (cuenca hidrográfica) trata el mapa de probabilidad invertido como un paisaje topográfico: las áreas de alta probabilidad (centros de los núcleos) son valles profundos, y las áreas de baja probabilidad (bordes) son crestas altas.6 El algoritmo simula una "inundación" desde "marcadores" (los centros de los núcleos) hasta que las cuencas de inundación de diferentes marcadores se encuentran. En estos puntos de encuentro, construye "presas", que se convierten en los límites que separan los objetos.19

#### 

#### 

#### 

#### 

#### 

#### **Implementación**

La implementación práctica de Watershed sobre la salida del U-Net implica una secuencia de pasos bien definidos que puedes implementar con bibliotecas como OpenCV y SciPy 18:

1. **Binarización y Limpieza:** Aplica un umbral al mapa de probabilidad para obtener una máscara binaria de primer plano. Luego, utiliza operaciones morfológicas para eliminar el ruido.55  
2. **Identificar Marcadores de Primer Plano Seguros:** Este es el paso más importante. Utiliza la **transformada de distancia** sobre la máscara de primer plano. Esta técnica calcula la distancia de cada píxel al fondo más cercano; los píxeles con una gran distancia están en el centro de los objetos. Al aplicar un umbral a este mapa de distancia, puedes aislar de manera confiable los centros de los núcleos, que servirán como tus marcadores.  
3. **Identificar la Región de Fondo y la Región Desconocida:** Dilata la máscara de primer plano para encontrar el área que es definitivamente fondo. La región que no es ni primer plano seguro ni fondo seguro es la región ambigua donde se encuentran los límites.  
4. **Etiquetar los Marcadores y Aplicar Watershed:** Asigna una etiqueta de entero única a cada marcador de primer plano. Luego, alimenta la imagen original y estos marcadores al algoritmo de Watershed. El algoritmo etiquetará cada región y marcará los límites, produciendo el mapa de segmentación de instancias final.

## 

## 

## 

## 

## 

## 

## 

## 

## 

## 

## **Parte V: Mejora de la Robustez \- Estrategias Avanzadas de Aumento de Datos**

El aumento de datos es una técnica indispensable en el aprendizaje profundo, especialmente con conjuntos de datos limitados o muy variables como los de imágenes biomédicas.56 Al crear versiones sintéticas pero realistas de los datos de entrenamiento, puedes mejorar significativamente la capacidad de generalización del modelo.

### **La Necesidad de Aumento en Microscopía**

Las imágenes de microscopía presentan desafíos únicos que hacen que el aumento de datos sea particularmente efectivo, como la variabilidad biológica en la forma y tamaño de las células, y la variabilidad técnica en la iluminación y la tinción.58 Un pipeline de aumento bien diseñado puede simular estas variaciones, enseñando al modelo a ser invariante a ellas.

### **Aumentos Específicos del Dominio**

Además de los aumentos estándar como giros y volteos, debes emplear técnicas más sofisticadas y específicas del dominio:

* **Transformaciones Geométricas:**  
  * **Rotaciones y Volteos:** Las células no tienen una orientación canónica, por lo que son aumentos seguros y efectivos.  
  * **Escalado y Recorte Aleatorio:** Simula variaciones en el aumento del microscopio.  
  * **Deformaciones Elásticas:** Esta es una técnica particularmente poderosa para imágenes biológicas, ya que simula las deformaciones no rígidas que ocurren en los tejidos blandos.  
* **Transformaciones Fotométricas (Color):**  
  * **Brillo, Contraste y Vibración de Color:** Simulan variaciones en la iluminación y la tinción, haciendo que el modelo sea más robusto.57  
  * **Ruido Gaussiano:** Simula el ruido del sensor de la cámara.

### 

### 

### **Implementación con Albumentations**

Para implementar un pipeline de aumento complejo, la biblioteca albumentations es la herramienta recomendada. Es altamente eficiente y, lo más importante, aplica la misma transformación geométrica tanto a la imagen como a su máscara correspondiente, asegurando que las etiquetas permanezcan alineadas. Tu objetivo es definir un pipeline de transformaciones que se aplicará a cada muestra durante el entrenamiento dentro de tu clase Dataset de PyTorch.

## 

## **Parte VI: La Exhibición \- Construyendo un Demostrador Interactivo con Streamlit**

Un modelo entrenado es una proeza técnica, pero una aplicación web interactiva que permite a otros usarlo es una demostración de impacto. **Streamlit** es una biblioteca de Python que te permite crear aplicaciones web para aprendizaje automático con un esfuerzo mínimo, sin necesidad de experiencia en desarrollo web.59

### **De Modelo a Aplicación**

Crear una aplicación de demostración hace que tu proyecto sea tangible y comprensible para una audiencia no técnica, y demuestra habilidades de despliegue. Streamlit es ideal por su simplicidad y su estrecha integración con el ecosistema de Python.59

### **Estructura de la Aplicación Streamlit**

Tu aplicación debe tener una estructura simple pero profesional 62:

1. **Título y Descripción:** Un encabezado claro que explique el propósito de la aplicación.  
2. **Carga de Archivos:** Un widget que permita al usuario cargar su propia imagen de microscopía.  
3. **Botón de Inferencia:** Un botón que inicie el proceso de segmentación.  
4. **Visualización de Resultados:** Una presentación comparativa que muestre la imagen original, el mapa de probabilidad del modelo y la segmentación de instancias final una al lado de la otra.

El flujo de trabajo de la aplicación debe ser el siguiente: cargar el modelo entrenado, gestionar la carga de la imagen del usuario, pre-procesarla de la misma manera que los datos de validación, ejecutar la inferencia del modelo, aplicar el post-procesamiento Watershed y, finalmente, visualizar los resultados. Esta aplicación interactiva sirve como la culminación perfecta del proyecto, cerrando el ciclo desde los datos crudos hasta un resultado final tangible y demostrable.

## **Conclusión: Sintetizando el Trabajo y Direcciones Futuras**

Haber completado este proyecto de extremo a extremo demuestra un conjunto de habilidades prácticas, teóricas y de resolución de problemas que son directamente aplicables a roles de alto rendimiento. Esta sección final proporciona una guía sobre cómo articular este trabajo y sugerencias para futuras exploraciones.

### **Articulando los Logros**

La presentación del proyecto es tan importante como su ejecución. Tu repositorio de GitHub debe tener un archivo README.md profesional que actúe como un informe técnico conciso. Debe incluir secciones sobre el resumen del problema 2, el conjunto de datos 20, la metodología (ingeniería de datos, arquitectura del modelo, entrenamiento destacando el uso de AMP 50, y post-procesamiento con Watershed 6), los resultados visuales y cuantitativos (IoU), y un enlace o instrucciones para ejecutar tu demostración interactiva.

### **Puntos Clave para Entrevistas**

Este proyecto te prepara para discutir una amplia gama de temas técnicos en una entrevista:

* **Gestión de Proyectos de Extremo a Extremo:** Describe cómo abordaste el proyecto desde la definición del problema hasta la demostración final.  
* **Optimización bajo Restricciones:** Explica por qué y cómo usaste AMP para entrenar en una GPU de 8 GB.  
* **Ingeniería de Datos Compleja:** Discute el desafío de las máscaras que se tocan y cómo diseñaste el pipeline de datos para resolverlo.  
* **Sistemas Híbridos de IA:** Explica la decisión de combinar un modelo de aprendizaje profundo con un algoritmo clásico de visión por computadora.  
* **Elección de Arquitectura y Función de Pérdida:** Justifica tus decisiones de diseño del modelo y la función de pérdida.

### **Trabajo Futuro**

Demostrar que has pensado en cómo extender el proyecto es una señal de curiosidad intelectual. Algunas posibles direcciones futuras incluyen:

* **Experimentar con Codificadores Pre-entrenados:** Reemplazar el codificador de la U-Net con un backbone pre-entrenado en ImageNet (como ResNet) para aprovechar el aprendizaje por transferencia.  
* **Despliegue como una API REST:** Envolver el modelo en una API utilizando Flask o FastAPI para demostrar habilidades de MLOps.  
* **Explorar el Pre-entrenamiento Auto-supervisado:** Utilizar un conjunto de datos más grande sin etiquetar para pre-entrenar el codificador, lo que podría mejorar la extracción de características.  
* **Cuantificación y Análisis Biológico:** Extender la aplicación para realizar análisis cuantitativos sobre las máscaras segmentadas (contar núcleos, calcular su área, etc.), conectando la salida del modelo con insights biológicos.

En resumen, este proyecto, cuando se ejecuta y presenta de manera exhaustiva, es una narrativa completa que demuestra competencia técnica, resolución de problemas pragmática y una comprensión profunda de cómo aplicar la IA para resolver desafíos significativos en un dominio de vanguardia.

#### **Works cited**

1. Artificial Intelligence Applications in Bioinformatics and Computational Biology by Ernest Bonat, Ph.D., accessed August 22, 2025, [https://ernest-bonat.medium.com/machine-learning-applications-in-genomics-life-sciences-by-ernest-bonat-ph-d-83598e67ccbc](https://ernest-bonat.medium.com/machine-learning-applications-in-genomics-life-sciences-by-ernest-bonat-ph-d-83598e67ccbc)  
2. Nucleus segmentation across imaging experiments: the 2018 Data Science Bowl \- PubMed, accessed August 22, 2025, [https://pubmed.ncbi.nlm.nih.gov/31636459/](https://pubmed.ncbi.nlm.nih.gov/31636459/)  
3. 11 Computer Vision Projects to Master Real-World Applications | DigitalOcean, accessed August 22, 2025, [https://www.digitalocean.com/resources/articles/computer-vision-projects](https://www.digitalocean.com/resources/articles/computer-vision-projects)  
4. Computer Vision in Pharmaceutical Quality Control: Vendors, Applications, and Trends \- IntuitionLabs, accessed August 22, 2025, [https://intuitionlabs.ai/pdfs/computer-vision-in-pharmaceutical-quality-control-vendors-applications-and-trends.pdf](https://intuitionlabs.ai/pdfs/computer-vision-in-pharmaceutical-quality-control-vendors-applications-and-trends.pdf)  
5. Semantic Segmentation — U-Net \- Medium, accessed August 22, 2025, [https://medium.com/@keremturgutlu/semantic-segmentation-u-net-part-1-d8d6f6005066](https://medium.com/@keremturgutlu/semantic-segmentation-u-net-part-1-d8d6f6005066)  
6. Watershed OpenCV \- PyImageSearch, accessed August 22, 2025, [https://pyimagesearch.com/2015/11/02/watershed-opencv/](https://pyimagesearch.com/2015/11/02/watershed-opencv/)  
7. Generative AI in the pharmaceutical industry: Moving from hype to reality \- McKinsey, accessed August 22, 2025, [https://www.mckinsey.com/industries/life-sciences/our-insights/generative-ai-in-the-pharmaceutical-industry-moving-from-hype-to-reality](https://www.mckinsey.com/industries/life-sciences/our-insights/generative-ai-in-the-pharmaceutical-industry-moving-from-hype-to-reality)  
8. AI-Powered Portfolio Management in Pharmaceuticals \- DrugPatentWatch, accessed August 22, 2025, [https://www.drugpatentwatch.com/blog/ai-powered-portfolio-management-in-pharmaceuticals/](https://www.drugpatentwatch.com/blog/ai-powered-portfolio-management-in-pharmaceuticals/)  
9. Training my first SDXL LoRA in Kohya\_ss on a RTX2070 it's been almost 4 days, is this typical for a low vram GPU? (Settings and specs in post) \- Reddit, accessed August 22, 2025, [https://www.reddit.com/r/StableDiffusion/comments/197js6u/training\_my\_first\_sdxl\_lora\_in\_kohya\_ss\_on\_a/](https://www.reddit.com/r/StableDiffusion/comments/197js6u/training_my_first_sdxl_lora_in_kohya_ss_on_a/)  
10. A Full Hardware Guide to Deep Learning \- Tim Dettmers, accessed August 22, 2025, [https://timdettmers.com/2018/12/16/deep-learning-hardware-guide/](https://timdettmers.com/2018/12/16/deep-learning-hardware-guide/)  
11. Community Test: Flux-1 LoRA/DoRA training on 8 GB VRAM using OneTrainer \- Reddit, accessed August 22, 2025, [https://www.reddit.com/r/StableDiffusion/comments/1fj6mj7/community\_test\_flux1\_loradora\_training\_on\_8\_gb/](https://www.reddit.com/r/StableDiffusion/comments/1fj6mj7/community_test_flux1_loradora_training_on_8_gb/)  
12. Thanks\! That rules them out completely EDIT: For \*training\* | Hacker News, accessed August 22, 2025, [https://news.ycombinator.com/item?id=38915042](https://news.ycombinator.com/item?id=38915042)  
13. Mastering U-Net: A Step-by-Step Guide to Segmentation from Scratch with PyTorch, accessed August 22, 2025, [https://medium.com/@fernandopalominocobo/mastering-u-net-a-step-by-step-guide-to-segmentation-from-scratch-with-pytorch-6a17c5916114](https://medium.com/@fernandopalominocobo/mastering-u-net-a-step-by-step-guide-to-segmentation-from-scratch-with-pytorch-6a17c5916114)  
14. U-Net A PyTorch Implementation in 60 lines of Code \- Aman Arora's Blog, accessed August 22, 2025, [https://amaarora.github.io/posts/2020-09-13-unet.html](https://amaarora.github.io/posts/2020-09-13-unet.html)  
15. How to Implement UNet in PyTorch for Image Segmentation from Scratch? \- Bhimraj Yadav, accessed August 22, 2025, [https://bhimraj.com.np/blog/pytorch-unet-image-segmentation-implementation](https://bhimraj.com.np/blog/pytorch-unet-image-segmentation-implementation)  
16. Data Science Bowl 2018 Kaggle competition | datasock, accessed August 22, 2025, [https://datasock.wordpress.com/2018/04/05/data-science-bowl-2018-kaggle-competition/](https://datasock.wordpress.com/2018/04/05/data-science-bowl-2018-kaggle-competition/)  
17. Loss Function Library \- Keras & PyTorch \- Kaggle, accessed August 22, 2025, [https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch](https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch)  
18. Image Segmentation with Watershed Algorithm \- OpenCV Documentation, accessed August 22, 2025, [https://docs.opencv.org/3.4/d3/db4/tutorial\_py\_watershed.html](https://docs.opencv.org/3.4/d3/db4/tutorial_py_watershed.html)  
19. Image Segmentation with Watershed Algorithm \- OpenCV Documentation, accessed August 22, 2025, [https://docs.opencv.org/4.x/d3/db4/tutorial\_py\_watershed.html](https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html)  
20. Kaggle 2018 Data Science Bowl | Broad Bioimage Benchmark Collection, accessed August 22, 2025, [https://bbbc.broadinstitute.org/BBBC038](https://bbbc.broadinstitute.org/BBBC038)  
21. 2018 Data Science Bowl \- Kaggle, accessed August 22, 2025, [https://www.kaggle.com/c/data-science-bowl-2018/data](https://www.kaggle.com/c/data-science-bowl-2018/data)  
22. Protein Structure Transformer \- arXiv, accessed August 22, 2025, [https://arxiv.org/pdf/2401.14819](https://arxiv.org/pdf/2401.14819)  
23. The transformative power of transformers in protein structure prediction \- PMC, accessed August 22, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10410766/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10410766/)  
24. Transformer-based deep learning for predicting protein properties in the life sciences \- PMC, accessed August 22, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9848389/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9848389/)  
25. \[D\] How much VRAM and RAM do I need for NLP transformer models? : r/MachineLearning, accessed August 22, 2025, [https://www.reddit.com/r/MachineLearning/comments/qrnozu/d\_how\_much\_vram\_and\_ram\_do\_i\_need\_for\_nlp/](https://www.reddit.com/r/MachineLearning/comments/qrnozu/d_how_much_vram_and_ram_do_i_need_for_nlp/)  
26. aqlaboratory/proteinnet: Standardized data set for machine learning of protein structure \- GitHub, accessed August 22, 2025, [https://github.com/aqlaboratory/proteinnet](https://github.com/aqlaboratory/proteinnet)  
27. ProteinNet: a standardized data set for machine learning of protein structure \- PMC, accessed August 22, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC6560865/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6560865/)  
28. Protein Secondary Structure Prediction with Long Short Term Memory Networks \- arXiv, accessed August 22, 2025, [https://arxiv.org/pdf/1412.7828](https://arxiv.org/pdf/1412.7828)  
29. OpenPose vs MediaPipe: Comprehensive Comparison & Analysis \- Saiwa, accessed August 22, 2025, [https://saiwa.ai/blog/openpose-vs-mediapipe/](https://saiwa.ai/blog/openpose-vs-mediapipe/)  
30. Key Point Extraction Using OpenPose (Left), MediaPipe Pose (Middle) and MMPose (Right) \- ResearchGate, accessed August 22, 2025, [https://www.researchgate.net/figure/Key-Point-Extraction-Using-OpenPose-Left-MediaPipe-Pose-Middle-and-MMPose-Right\_fig2\_366303399](https://www.researchgate.net/figure/Key-Point-Extraction-Using-OpenPose-Left-MediaPipe-Pose-Middle-and-MMPose-Right_fig2_366303399)  
31. A comprehensive analysis of the machine learning pose estimation models used in human movement and posture analyses: A narrative review, accessed August 22, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11566680/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11566680/)  
32. Custom Dataset — OpenPifPaf Guide, accessed August 22, 2025, [https://openpifpaf.github.io/plugins\_custom.html](https://openpifpaf.github.io/plugins_custom.html)  
33. Pose Estimation \- Ultralytics YOLO Docs, accessed August 22, 2025, [https://docs.ultralytics.com/tasks/pose/](https://docs.ultralytics.com/tasks/pose/)  
34. How to Train a Custom YOLOv8 Pose Estimation Model \- Roboflow Blog, accessed August 22, 2025, [https://blog.roboflow.com/train-a-custom-yolov8-pose-estimation-model/](https://blog.roboflow.com/train-a-custom-yolov8-pose-estimation-model/)  
35. Kaggle Data Science Bowl 2018 : Find and segment nuclei \- Page 2 \- Deep Learning, accessed August 22, 2025, [https://forums.fast.ai/t/kaggle-data-science-bowl-2018-find-and-segment-nuclei/9966?page=2](https://forums.fast.ai/t/kaggle-data-science-bowl-2018-find-and-segment-nuclei/9966?page=2)  
36. EIDOSLAB/torchstain: Stain normalization tools for histological analysis and computational pathology \- GitHub, accessed August 22, 2025, [https://github.com/EIDOSLAB/torchstain](https://github.com/EIDOSLAB/torchstain)  
37. Writing Custom Datasets, DataLoaders and Transforms \- PyTorch documentation, accessed August 22, 2025, [https://docs.pytorch.org/tutorials/beginner/data\_loading\_tutorial.html](https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html)  
38. usuyama/pytorch-unet: Simple PyTorch implementations of U-Net/FullyConvNet (FCN) for image segmentation \- GitHub, accessed August 22, 2025, [https://github.com/usuyama/pytorch-unet](https://github.com/usuyama/pytorch-unet)  
39. U-Net for brain MRI \- PyTorch, accessed August 22, 2025, [https://pytorch.org/hub/mateuszbuda\_brain-segmentation-pytorch\_unet/](https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/)  
40. U-Net for image segmentation, PyTorch implementation. \- GitHub, accessed August 22, 2025, [https://github.com/hiyouga/Image-Segmentation-PyTorch](https://github.com/hiyouga/Image-Segmentation-PyTorch)  
41. Simple pytorch implementation of the u-net model for image segmentation \- GitHub, accessed August 22, 2025, [https://github.com/clemkoa/u-net](https://github.com/clemkoa/u-net)  
42. milesial/Pytorch-UNet: PyTorch implementation of the U-Net for image semantic segmentation with high quality images \- GitHub, accessed August 22, 2025, [https://github.com/milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)  
43. Attention UNET in PyTorch \- Idiot Developer, accessed August 22, 2025, [https://idiotdeveloper.com/attention-unet-in-pytorch/](https://idiotdeveloper.com/attention-unet-in-pytorch/)  
44. sfczekalski/attention\_unet: Neural Network for semantic segmentation \- GitHub, accessed August 22, 2025, [https://github.com/sfczekalski/attention\_unet](https://github.com/sfczekalski/attention_unet)  
45. Implementation of dice loss — vision — PyTorch | by Hey Amit | Data Scientist's Diary, accessed August 22, 2025, [https://medium.com/data-scientists-diary/implementation-of-dice-loss-vision-pytorch-7eef1e438f68](https://medium.com/data-scientists-diary/implementation-of-dice-loss-vision-pytorch-7eef1e438f68)  
46. matin-ghorbani/IoU-from-Scratch: Intersection Over Union from Scratch using PyTorch \- GitHub, accessed August 22, 2025, [https://github.com/matin-ghorbani/IoU-from-Scratch](https://github.com/matin-ghorbani/IoU-from-Scratch)  
47. What is Intersection over Union (IoU)? | Definition \- Encord, accessed August 22, 2025, [https://encord.com/glossary/iou-definition/](https://encord.com/glossary/iou-definition/)  
48. PyTorch-ENet/metric/iou.py at master \- GitHub, accessed August 22, 2025, [https://github.com/davidtvs/PyTorch-ENet/blob/master/metric/iou.py](https://github.com/davidtvs/PyTorch-ENet/blob/master/metric/iou.py)  
49. Mean Intersection over Union (mIoU) — PyTorch-Metrics 1.8.1 documentation \- Lightning AI, accessed August 22, 2025, [https://lightning.ai/docs/torchmetrics/stable/segmentation/mean\_iou.html](https://lightning.ai/docs/torchmetrics/stable/segmentation/mean_iou.html)  
50. Automatic Mixed Precision — PyTorch Tutorials 2.8.0+cu128 ..., accessed August 22, 2025, [https://docs.pytorch.org/tutorials/recipes/recipes/amp\_recipe.html](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)  
51. Train With Mixed Precision \- NVIDIA Docs, accessed August 22, 2025, [https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)  
52. Automatic Mixed Precision examples — PyTorch 2.8 documentation, accessed August 22, 2025, [https://docs.pytorch.org/docs/stable/notes/amp\_examples.html](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)  
53. Supercharging PyTorch Training: 10 GPU Optimizations with Functional Code \- Medium, accessed August 22, 2025, [https://medium.com/@singh.tarus/supercharging-pytorch-training-10-gpu-optimizations-with-functional-code-f50b8f719bad](https://medium.com/@singh.tarus/supercharging-pytorch-training-10-gpu-optimizations-with-functional-code-f50b8f719bad)  
54. Image Segmentation with Watershed Algorithm \- OpenCV Python \- GeeksforGeeks, accessed August 22, 2025, [https://www.geeksforgeeks.org/computer-vision/image-segmentation-with-watershed-algorithm-opencv-python/](https://www.geeksforgeeks.org/computer-vision/image-segmentation-with-watershed-algorithm-opencv-python/)  
55. Image Segmentation in OpenCV Python with Watershed Algorithm \- Tutorialspoint, accessed August 22, 2025, [https://www.tutorialspoint.com/image-segmentation-in-opencv-python-with-watershed-algorithm](https://www.tutorialspoint.com/image-segmentation-in-opencv-python-with-watershed-algorithm)  
56. Test-time augmentation for deep learning-based cell segmentation on microscopy images, accessed August 22, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7081314/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7081314/)  
57. Review of Image Augmentation Used in Deep Learning-Based Material Microscopic Image Segmentation \- MDPI, accessed August 22, 2025, [https://www.mdpi.com/2076-3417/13/11/6478](https://www.mdpi.com/2076-3417/13/11/6478)  
58. Semantic Aware Data Augmentation for Cell Nuclei Microscopical Images With Artificial Neural Networks \- CVF Open Access, accessed August 22, 2025, [https://openaccess.thecvf.com/content/ICCV2021/papers/Naghizadeh\_Semantic\_Aware\_Data\_Augmentation\_for\_Cell\_Nuclei\_Microscopical\_Images\_With\_ICCV\_2021\_paper.pdf](https://openaccess.thecvf.com/content/ICCV2021/papers/Naghizadeh_Semantic_Aware_Data_Augmentation_for_Cell_Nuclei_Microscopical_Images_With_ICCV_2021_paper.pdf)  
59. Streamlit Python: Tutorial \- DataCamp, accessed August 22, 2025, [https://www.datacamp.com/tutorial/streamlit](https://www.datacamp.com/tutorial/streamlit)  
60. Building an Image Recognition Application \- Snowflake, accessed August 22, 2025, [https://www.snowflake.com/en/developers/solutions-center/building-an-image-recognition-application-using-streamlit-snowpark-pytorch-and-openai/](https://www.snowflake.com/en/developers/solutions-center/building-an-image-recognition-application-using-streamlit-snowpark-pytorch-and-openai/)  
61. Image segmentation: Train a U-Net model to segment brain tumors ..., accessed August 22, 2025, [https://blog.ovhcloud.com/image-segmentation-train-a-u-net-model-to-segment-brain-tumors/](https://blog.ovhcloud.com/image-segmentation-train-a-u-net-model-to-segment-brain-tumors/)  
62. PyTorch Archives \- \- OVHcloud Blog, accessed August 22, 2025, [https://blog.ovhcloud.com/tag/pytorch/](https://blog.ovhcloud.com/tag/pytorch/)  
63. Building a Real-Time Image Classifier with Streamlit and PyTorch || PART 1 \- YouTube, accessed August 22, 2025, [https://www.youtube.com/watch?v=IBiL04fVLD8](https://www.youtube.com/watch?v=IBiL04fVLD8)