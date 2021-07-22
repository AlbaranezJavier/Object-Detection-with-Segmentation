# Trabajo Fin de Máster
Título: Detección de objetos en imágenes con redes de segmentación.

Máster: Visión Artificial.

Universidad: Rey Juan Carlos.

Mes/Año: 07/2021.

### Temática y objetivo

Soy miembro del equipo de competición [UMotorsport](http://u-motorsport.com/2019/08/19/umotorsport/) y este equipo
colabora con el grupo de investigación [CAPO](http://caporesearch.es/), al cual también pertenezco. La competición internacional
en la que se encuentra envuelto el proyecto es [Formula Student](https://www.formulastudent.es/), el cual tiene por objetivo
dotar a un coche de carreras de una inteligencia artificial capaz de conducir de forma autónoma por un circuito delimitado por balizas (conos).

![conos](https://user-images.githubusercontent.com/32954090/126610850-411602ff-a093-4063-9238-94b59b7b82a6.PNG)

En este proyecto se propone un modelo de red neuronal profunda, llamado SNet-3L (Segmentation Network - 3 Levels), que propone una forma distinta de afrontar la detección de objetos, realizando una hibridación de conceptos y proponiendo criterios para establecer los hiperparametros que definen la red neuronal (NN, Neural Network). Para realizar la detección de objetos se utiliza la segmentación semántica de objetos en vez de las anclas, realizando un postprocesado sobre los datos obtenidos para delimitar y localizar los objetos clasificados anteriormente a nivel de píxel.

![esquema inferencia](https://user-images.githubusercontent.com/32954090/126610995-bf47a5b3-d0cf-4a85-a72a-a896338fa5e7.PNG)
    
Finalmente, se compara SNet-3L con el modelo de referencia en el estado del arte, EfficientDet-D0 (D0). Los resultados experimentales muestran que: los tiempos de entrenamiento, el número de parámetros que define la red, el tiempo de inferencia y la precisión de los resultados, mejoran con respecto D0. Además, el entrenamiento de ambos modelos con datos mixtos (sintéticos 899 y reales 242), parecen ser suficientes para esta aplicación.

### Antecedentes

Tengo otros dos proyectos muy relacionados con el objetivo de este: (i) [un generador de imágenes sintéticas de conos](https://github.com/AlbaranezJavier/SyntheticConeDatasetGenerator) y (ii) [un simulador de conducción](https://github.com/AlbaranezJavier/UnityTrainerPy). Ambos proyectos se encuentran adaptados a este problema y hacen uso del motor gráfico de [Unity](https://unity.com/es).

### Enlaces de interés

| [Memoria](https://github.com/AlbaranezJavier/JaviProject/edit/main/docs/TFM/TFM_JAM.pdf) |

## Estructura del código

- Data -> gestiona todo lo referido con los datos de entrenamiento
  + DataManager.py -> esta clase permite cargar los conjunto de datos para: extraer las etiquetas de un json, partir en conjuntos de entrenamiento y validación, generación de lotes para el entrenamiento, etc.
  + Jsons2json.py -> este script permite unir varios json en uno solo, útil cuando se va etiquetando un conjunto de datos poco a poco.
  + seg2bbox_csv.py -> permite traducir etiquetas generadas por segmentación a cajas contenedoras, guardándolas en un csv.
- Logs -> esta carpeta contiene los registros de entrenamiento para cada modelo y versión.
- Models -> almacena versiones optimizadas de modelos entrenados.
- Networks -> almacena las arquitecturas neuronales y sus esquemas realizados con DrawIO.
  + HNet.py
  + hnet_0.drawio
  + SNet.py
  + snet.drawio
- Perception -> módulo en desarollo
- Statistics -> contiene todos los scripts necesarios para analizar las métricas del proyecto.
  + inference_stats.py -> genera las métricas para probar cómo de bueno es un modelo.
  + Metrics.py -> esta clase contiene todas las métricas implementadas.
  + StatsData.py -> analiza el conjunto de datos seleccionado.
  + StatsModel.py -> prepara el modelo de la red neuronal para gestionar las métricas.
  + training_stats.py -> permite extraer las gráficas de entrenamiento leyendo los json almacenados en /Logs
- Tools -> herramientas empleadas en el proyecto.
  + get_set4json.py -> recoge las etiquetas de un json y genera un json con el conjunto de entrenamiento y validación.
- Weights -> almacena los pesos en función del modelo.
- inference_net.py -> permite generar etiquetas de las imágenes pasadas como argumento con un modelo neuronal.
- requirements.txt -> contiene todas las dependencias necesarias para ejecutar el proyecto.
- training_net.py -> permite entrenar cualquiera de los modelos desarrollados.

## Datos
Durante el proyecto se han generado dos tipos de datos: sintéticos y reales. En la siguiente figura se pueden ver todos los datos extraídos del conjunto de datos completo.![datos](https://user-images.githubusercontent.com/32954090/126609008-1c68ecf5-21a7-4c81-997b-a512ec5bd77f.PNG)

### Ejemplos reales
![ejemplos reales](https://user-images.githubusercontent.com/32954090/126608620-299564ae-1c41-4cc1-97b5-bddac35a6db7.PNG)

### Ejemplos sintéticos
![ejemplos sinteticos](https://user-images.githubusercontent.com/32954090/126608622-9fca332f-0e3f-4852-90f5-cab98bb811c6.PNG)

## Arquitectura
Se ha generado una red neuronal de segmentación con una arquitectura escalable denominada SNet. 

![snet3l](https://user-images.githubusercontent.com/32954090/126610466-3f450d23-3fad-4523-b8f6-8492ee625194.png)

La creación del sistema de detección propuesto en este TFM, ha sido muy influenciado por:
- Arquitecturas tipo [U-net](https://arxiv.org/abs/1505.04597), [Autoencoders](https://www.aaai.org/Papers/AAAI/1987/AAAI87-050.pdf) y [Redes Neuronales Convolucionales (CNNs, Convolutional Neural Networks)](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf). El modelo final utiliza este tipo de arquitecturas para procesar la imagen a múltiples resoluciones.
- [Combinación de pirámides espaciales, (SPP, Spatial Pyramid Pooling)](https://arxiv.org/abs/1406.4729) y [MG (Multi Grid)](https://arxiv.org/abs/1901.10415) para aplicar otro tipo de restrictores y prolongadores dimensionales, en vez de \textit{maxpool} y \textit{upsampling}.
- Técnicas de procesamiento de imagen ([pirámides Laplacianas](https://arxiv.org/abs/1605.02264)), lo que ha motivado la idea de introducir la información de entrada a los distintos niveles que componen la arquitectura neuronal.

## Resultados
El modelo con el que me he comparado en este proyecto es EfficientDet-D0 (D0), este ha sido reentrenado con el mismo conjunto de datos que SNet (utilizar transfer learning no mejoraba los resultados en este caso) y optimizado para el mismo hardware con TensorRT.

### Resultados comparando detecciones delimitadas por cajas contenedoras (bbox)
La imagen de abajo, muestra como la versión equivalente a D0 es SNet4L y esta sigue siendo mucho más rápida que D0.

![image](https://user-images.githubusercontent.com/32954090/126611342-05c535f6-a54b-49ca-8343-6d5d7c7239b4.png)

Pero si comparamos las mismas métricas para un problema de segmentación (imagen siguiente), se puede apreciar como el modelo propuesto es muy superior en todas las versiones.

![image](https://user-images.githubusercontent.com/32954090/126611742-fd6b909f-5f2a-41fb-82e0-ad3307d3d660.png)

#### Detecciones, el orden de representación es rojo (etiqueta), verde (SNet-3L) y azul (D0)
![detecciones](https://user-images.githubusercontent.com/32954090/126614294-902cdf93-41ac-4fbe-aa8c-5b26fdad70a8.PNG)

### Detecciones realizadas por SNet-3L, entrenada con datos sintéticos, reales y mixtos.
![segmentation](https://user-images.githubusercontent.com/32954090/126614923-7c9fba9a-e942-48af-85eb-1034da23f438.PNG)


## Conclusiones
Para este conjunto de datos se podría decir que:
- Utilizar los resultados de segmentación es mucho más acertado que emplear las anclas típicas en detección, pues permiten delimitar mucho mejor los límites del objeto en la imágen.
- Se ha superado el estado del arte, utilizando un sistema de detección de dos estados invertido (segmentación + postprocesado) contra uno de un estado (EfficientDet-D0).
- Se ha desarrollado una red de detección de balizas capaz de operar en tiempo real y con una estructura fácilmente escalable.
- El sistema desarrollado es mucho más simple, eficiente (en cuanto el número de neuronas) y rápido de entrenar que los modelos de detección estándar.

## Trabajos futuros
Los resultados son prometedores, pero es necesario explorar esta solución en un problema estándar y comparar los resultados ahí. Por otro lado, esta red tiene el inconveniente de que cuando dos objetos de la misma clase se superponen, el sistema los detecta como un solo objeto. La arquitectura propuesta queda lejos de emplear todos los avances propuestos en el estado del arte, por lo que su capacidad de mejora sigue siendo grande.

## Crédito
Quiero agraceder el apoyo recibido por mis tutores y compañeros del grupo CAPO, en especial a mis compañeras Laura Llopis Ibor y Susana Pineda de Luelmo.

| [JaviProject web](https://albaranezjavier.github.io/JaviProject/) |


