import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pickle
from scipy.stats import zscore
from scipy.signal import find_peaks, butter, sosfilt

#Funciones relacionadas al analisis inicial de la Frecuencia Basal de la DOE

def deteccion_picos(duracion: int, z_score: np.array, umbral: float, distancia: int):
    """
    Esta funcion detecta los picos de la DOE para un determinado umbral y distancia. El umbral se determinara previamente observando la senal electrica.
    Entradas: 
        duracion (int) = duracion en segundos de cada registro electrico (cada archivo)
        z_score (np.array) = numpy array conteniendo el zscore de la senal electrica
        umbral (int) = valor de umbral de deteccion
        distancia (int) = duracion de la DOE, esta puede cambiar segun la especie. Este parametro determinara que no se detecte una misma DOE mas de una vez. Para G. omarorum se recomienda que distancia = 150
    Salidas:
        EOD_frequencies (np.array) = np array conteniendo las frecuencias instantaneas de la DOE para la senal brindada
        EOD_peaks (np.array) = np array con los tiempos (en indices) donde se da cada DOE

    """
    EODTime = np.linspace(0, duracion, len(z_score)) #duracion en s
    EOD_peaks, _ = find_peaks(z_score, height = umbral, distance=distancia)
        #calculamos los intervalos y frecuencias
    EOD_intervals = np.diff(EODTime[EOD_peaks])
    EOD_frequencies = [1/j for j in EOD_intervals]

    return EOD_frequencies, EOD_peaks

def preprocesar_EOD(EOD: np.array, fm: int):
    """
    Este es un paso opcional en el procesamiento. Implica alinear los canales al 0 para asegurar no perder ninguna senal, y un filtro de banda.

    Entradas:
        EOD = registro electrico (un unico canal)
        fm = frecuencia de muestreo mediante la cual se obtuvo el registro

    Salida:
        Senal preprocesada
    """
    EOD -= np.median(EOD)
    sos = butter(3, [300,3000], btype='bandpass', fs=fm, output='sos')
    if len(EOD) > 0:
        EOD = sosfilt(sos, EOD)
    else:
        print('EOD shape 0')
    return EOD

def FB_DOE_analisis(data_folder: str, fm: int, duracion: int, nombre_guardar: str, distancia=150, n_canales=2, preprocesar=True):
    """
    Esta funcion tiene como finalidad obtener la frecuencia instantanea de la DOE para todos los archivos de registro de un mismo 
    experimento.
    Entradas: 
        data_folder (str): carpeta donde se encuentran los archivos '.bin' con los registros electricos
        fm (int): frecuencia de muestreo mediante la cual se obtuvieron los registros
        duracion (int): duracion de cada archivo de registro expresado en segundos
        nombre_guardar (str): nombre del archivo '.pkl' donde se guardaran las FB-DOE. Recordar agregar '.pkl' al final
        distancia (int) : duracion de la DOE, esta puede cambiar segun la especie. Este parametro determinara que no se detecte 
                            una misma DOE mas de una vez. Para G. omarorum se recomienda que distancia = 150
        n_canales (int) : numero de canales que se utilizaron para registrar. Default: 2
        preprocesar (bool) : determina si se preprocesa la senal o no. Default: True
    
    Salidas: 
        Guarda en la carpeta brindada (data_folder) un archivo '.pkl' que contiene un diccionario. Este diccionario contiene las 
        frecuencias instantaneas de la DOE y los tiempos (en indices) en los cuales se da cada descarga. 
        El diccionario contiene dos sub-diccionarios 'FB-DOE' y 'Peak-time'. En cada uno de ellos se guarda bajo una clave que indica
        el nombre del archivo correspondiente, la frecuencia instantanea y tiempos de descarga, respectivamente.
    """
    os.chdir(data_folder)
    archivos = sorted(glob.glob( '*.bin'))
    print('Hay ' + str(len(archivos)) + ' archivos')

    #creamos el diccionario vacio
    fish = {
            'FB-DOE': {},
            'Peak-time': {}}

    for k, archivo in enumerate(archivos):
        EOD = np.fromfile(archivo,dtype=np.float64) #cargamos el archivo
        EOD_ch = EOD.reshape((int(EOD.shape[0]/n_canales),n_canales))
    
        if preprocesar:
            #Pre-procesamos para nivelar el ruido
            for column in range(EOD_ch.shape[1]):
                EOD_ch[:,column] = preprocesar_EOD(EOD_ch[:,column], fm) 

        # combinamos los  canales sumando los cuadrados de los mismos
        for i in range(EOD_ch.shape[1]-1):
            if i == 0:
                EOD = np.square(EOD_ch[:,i], dtype=np.float64)
            else:
                EOD += np.square(EOD_ch[:,i], dtype=np.float64)
        
        # calculamos el z-score
        z_score = zscore(EOD)

        if k == 0:
            plt.figure()
            plt.plot(z_score[:150000])
            plt.show()
            umbral = input("Introduzca el umbral para la deteccion de DOEs: ")

        # detectamos picos y generamos el vector de tiempo
        EOD_frequencies, EOD_peaks = deteccion_picos(duracion=duracion, z_score=z_score, umbral=float(umbral), distancia=distancia)
        
        #guardamos
        name = archivo[10:-4] #esto puede cambiar 
        fish['FB-DOE'][name] = EOD_frequencies
        fish['Peak-time'][name] = EOD_peaks
        print('Termino archivo ' + str(k))
    
    with open(nombre_guardar, 'wb') as fp: #cambiar nombre a nombre deseado del archivo a guardar
        pickle.dump(fish, fp)
    
    

def obtener_FBDOE_media(archivo_pez: dict, duracion: int, graficar=True):
    """
    Esta funcion calcula y grafica la mediana y desvio para cada archivo de registro electrico. 
    Entradas:
        archivo_pez (dict): diccionario conteniendo la FB-doe del pez para cada archivo de registro. Obtenido de FB_DOE_analysis.
        graficar (bool): determina si graficar la frecuencia media en el tiempo. Default True. La figura se guardara en la carpeta
                        donde se este trabajando

    Salidas:
        Means (list): lista que contiene la FB-doe mediana para cada archivo brindado, ordenadas en el tiempo
        Desvio (list): lista que contiene el desvio estandar para cada archivo brindado, ordenadas en el tiempo
    """
    Means = []
    Desvio = []
    for key in sorted(list(archivo_pez['FB-DOE'].keys())):
        mean = np.median(archivo_pez['FB-DOE'][key])
        std = np.std(archivo_pez['FB-DOE'][key])
        Means.append(mean)
        Desvio.append(std)

    if graficar:
        x = np.linspace(0, duracion*len(Means), num=len(Means))
        fig, ax = plt.subplots()
        ax.plot(x,Means, c='k')
        ax.scatter(x, Means, c='k')
        ax.fill_between(x, Means-std, Means+std, alpha=.5)
        ax.set(xlabel='Tiempo (min)', ylabel='FB-DOE mediana (Hz)')
        fig.savefig('FB_DOE_mediana.svg', format='svg')
        plt.show()

    return Means, Desvio

def detectar_eventos(EOD_zscore: np.array, EOD_peak_time: np.array, umbral_evento: float):
        """
        Esta funcion detecta eventos de alta frecuencia en el registro electrico. Detecta un evento si el zscore de la FB-DOE 
        supera cierto umbral y lo sostiene por al menos 3 DOEs consecutivas. 

        Entradas:
            EOD_zscore (np.array): contiene valores de zscore de la FB-DOE
            EOD_peak_time (np.array): tiempos correspondientes a los zscore brindados donde se dan los picos 
            umbral_eventos (float): valor umbral para definir una frecuencia instantanea como de alta frecuencia. Por referencia, 
                                    1.5 equivale a tomar valores de FB-DOE mayores a la media mas 1.5 desvios estandar. 

        Salidas:
            i_novel (np.array): np array que contiene los indices donde se da un evento de alta frecuencia (primer DOE)
            novel (np.array): valores de zscore correspondientes al comiento de cada evento (correspondiente a cada i_novel)
        """
        events =[]
        for k in np.arange(1,len(EOD_zscore)-3):
            is_event = EOD_zscore[k] > umbral_evento and EOD_zscore[k+1] > umbral_evento and EOD_zscore[k+2] > umbral_evento and EOD_zscore[k-1] < umbral_evento
            if is_event: 
                events.append(k) 
            i_novel = EOD_peak_time[[k for k in events]]
            novel = EOD_zscore[events]
        
        return i_novel, novel
            
