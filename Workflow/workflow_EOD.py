import pandas as pd
import numpy as np
import glob
import os
import pickle
import re
from scipy.spatial import distance
from datetime import datetime
##Nos movemos a la carpeta con las funciones (no cambiar!!)
folder_funciones = "C:\Users\Compras\Documents\Analisis Estacion\Funciones"
os.chdir(folder_funciones)
from EOD_analysis import * #esta linea importa todas las funciones definidas en el modulo 'locomotor analisis'
from Locomotor_analisis import *

### input aca
num_individuos = 6
path_datos = 'D:\\datos_GPetersii\\datos_GPetersii\\'
fm = 10000 #frecuencia de muestreo
duracion = 3600 #duracion de los registros en s

## A continuacion se ejecuta el analisis general de registros de la estacion conductual a partir de la carpeta
for i in range(1,num_individuos):
    directorio = path_datos + 'Fish' + str(i) + '\\raw'
    nombre = 'fish' + str(i) + '_FB_DOE.pkl'
    FB_DOE_analisis(data_folder=directorio, fm=fm, duracion=duracion, nombre_guardar=nombre, distancia=300, n_canales=2)

