# Electric fish analysis
## Introducción
Este repositorio contiene scripts para analizar los datos obtenidos de la estacion conductual de peces eléctricos de Facultad de Ciencias. Contiene analisis exploratorios asi como más generales para los registros eléctricos y el trackeo realizado por DeepLabCut sobre los videos.
Cada archivo está comentado y con una breve introduccion al comienzo. Los detalles de funcionamiento de cada script se encontrara en los comentarios, en este documento hacemos una recorrida general por el repositorio.  

A su vez contiene scripts utilizados para el analisis de datos conductuales obtenidos en el laboratorio de Nate Sawtell (Columbia University) de individuos de la especie Gnathonemus Petersii (carpeta Gnathonemus Petersii). Estos scripts estan comentados pero no seran descritos en esta guia. 

El siguiente diagrama esquematiza el funcionamiento de este repositorio y el tipo de datos que toma y que devuelve:
 <img src="/images/diagrama.png" alt="structure_repo" width="800"/>
 
 Como se muestra en el diagrama, los datos de la estación conductual son obtenidos mediante Bonsai rx. (Lopes et al, 2015)[^1]. El workflow correspondiente se encuentra en la carpeta "registro". De esta manera se obtienen archivos de video en formato ".avi" y archivos numericos que contienen el registro eléctrico en formato ".bin". Los archivos de registro eléctrico pasan directamente a ser analizados por codigo de este repositorio (en el esquema: todo lo que se encuentra dentro del recuadro rojo). Los videos son procesados inicialmente por DeepLabCut (Mathis et al., 2018)[^2] , por un modelo entrenado por la autora de este repositorio con datos generados para su tesis de maestria (carpeta "DeepLabCut"). Se aconseja para futuros experimentos entrenar un nuevo modelo con los videos generados en ese momento, para asegurar un buen seguimiento. Luego de trackear al animal utilizando DeepLabCut (u otro sistema de tracking), los archivos ".h5" con las posiciones de cada punto seran utilizados para analisis generales, como los que se describen en esta guia y tutorial; asi como para otros analisis que no se describen pero se puede encontrar ejemplos en la carpeta de G. petersii.
 
A continuación se detalla la funcionalidad de cada archivo y un sugerido workflow para comenzar el analisis de datos conductuales de la estacion. 

## Funcionalidades
### Pre-Procesamiento
Este repositorio incluye codigo de "evaluación" y de "limpieza". 

El codigo de **evaluación** son cuadernos de jupyter organizados en celdas para un mejor troubleshooting de los análisis asi como la organizacion de los datos. Tenemos dos rutinas de este tipo, que se describen a continuación.  

   1. Exploracion_DOE.ipynb: 
                Este rutina lee un archivo '.bin' conteniendo registro eléctrico y otros parametros como el numero de canales de registro (2) y la frecuencia de muestreo (definida según el experimento). Luego, a lo largo de las celdas se irá analizando el registro eléctrico para evaluar la relación señal-ruido y, ultimamente, determinar el umbral a utilizar para detectar las descargas del órgano eléctrico para un determinado pez. Al final encuentra los picos en el registro para un determinado umbral y grafica la Frecuencia instantánea en función del tiempo para corroborar la correcta detección.
            
   2. Exploracion_DLC.ipynb:
               Esta rutina toma un archivo '.h5', salida de DeepLabCut y otros parámetros como el número de partes del cuerpo trackeadas y el numero de frames por segundo a las cuales se registra el video. 

El código de **limpieza** tiene la funcionalidad de suavizar el trackeo generado por DeepLabCut. En la carpeta "Limpieza" se encuentran dos rutinas, con la misma funcionalidad pero a escalas distintas.

   1. limpiar_pose.ipynb:
               Este codigo permite estudiar el funcionamiento del suavizado para un solo video trackeado. Toma un archivo '.h5' y grafica las posiciones en x y en y de todas las partes del cuerpo, posterior al suavizado, para corroborar que no haya saltos inesperados en la posición. El suavizado consiste en eliminar todos los datos donde la 'likelihood' (o probabilidad) de una parte del cuerpo es menor a 0.99, y luego interpolar mediante sus valores vecinos la posición de esa parte del cuerpo en esa frame. 
   
   2. limpiar_pose.py:
               Este código tiene la misma funcionalidad que el anterior, pero realiza el suavizado para todos los archivos de trackeo en una determinada carpeta. Aquí se puede brindar carpetas por pez o una carpeta conteniendo todos los archivos '.h5' del experimento. La salida es la generación de los archivos '.h5' actualizados, con la adición del sufijo '_clean', los cuales se guardarán en la misma carpeta. 

### Análisis

Estas rutinas generan la estructura de datos que analizaremos a partir de los datos crudos. 

   1. _Adquisicion_FBDOE.py:_ 
               Este script tiene la misma funcionalidad que 'Exploracion_DOE.ipynb' pero para todos los archivos en una carpeta. Genera un diccionario con la FB-DOE y el tiempo de cada pico. 
   El diccionario que guarda es un diccionario que contiene dos diccionarios: FB-DOE y Peak-time. Cada uno de estos diccionarios cuenta con un elemento por archivo cuya key es el nombre del archivo y el valor es un vector de FB-DOE y Peak-Time de cada archivo, respectivamete. Luego, este diccionario se utiliza en otros scripts para no re-detectar DOEs ni re-calcular la frecuencia basal. 
                Por cada pez se tendrá un archivo '.pkl' salido de este script que guarda el diccionario generado. 


      La estructura del archivo .pkl es la siguiente
      
        <img src="/images/data.png" alt="data_structure_EOD" width="400"/>

   2. _Adquisicion_locomocion.py:_
               Este script genera el analisis inicial de la locomocion. Para esto suaviza el trackeo y calcula el tiempo en movimiento del animal a lo largo del registro brindado. A su vez genera los mapas de distribucion espacial del tiempo de visita, frecuencia de descarga y densidad de muestreo. 
                

### Post-procesamiento
Una vez se obtiene el archivo pkl de actividad electrica y el analisis locomotor inicial, se puede seguir con otros procesamientos. Esto incluye la evaluacion de los modelos entrenados, analisis estadisticos de variables relevantes, etc. En este repositorio se provee un script para analizar el rendimiento de DeepLabCut (eval_DLC.py). Ejemplos de otros procesamientos, como la segmentacion de unidades comportamentales se pueden acceder en la carpeta G. petersii

## Workflow sugerido

Una vez obtenidos los archivos de video y de registro eléctrico de la estación conductual, se sugiere comenzar con el análisis de la Frecuencia Basal de la Descarga del Organo Eléctrico (FBDOE). Para esto, se recomienda comenzar con la exploración ("Exploracion_DOE.ipynb") para cada pez. Luego de explorar los registros para cada pez y determinar los umbrales apropiados para la detección de las descargas, pasar a "Adquisicion_FBDOE.py". 

Para el analisis de la actividad locomotora, se deberia obtener el segumiento del animal utilizando algun sistema de tracking. Si se utiliza DeepLabCut, se puede pasar directamente a utilizar el código aquí publicado. En caso de utilizar otro sistema de tracking, el código aqui presente ha de ser modificado como sea apropiado. 



[^1]: Lopes, G., Bonacchi, N., Frazão, J., Neto, J. P., Atallah, B. V., Soares, S., ... & Kampff, A. R. Bonsai: an event-based framework for processing and controlling data streams. Frontiers in neuroinformatics, 9, 7 (2015). https://doi.org/10.3389/fninf.2015.00007

[^2]: Mathis, A., Mamidanna, P., Cury, K.M. et al. DeepLabCut: markerless pose estimation of user-defined body parts with deep learning. Nat Neurosci 21, 1281–1289 (2018). https://doi.org/10.1038/s41593-018-0209-y
