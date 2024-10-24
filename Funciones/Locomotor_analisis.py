import pandas as pd
import numpy as np
import glob
import os
from scipy.spatial import distance
from datetime import datetime
import matplotlib.pyplot as plt

def clean(dir: str, n_keypoints: int, umbral_likelihood=.99):
    """
    Esta funcion se encarga de suavizar el trackeo de DeepLabCut. Para esto chequea todas las likelihoods de cada keypoint para cada frame, y en el caso en que la
    likelihood sea menor que un cierto valor (default: 0.99) elimina esa posicion (solo para ese keypoint). Luego se interpolan linealmente los valores 
    eliminados segun los valores previos y siguientes para ese keypoint. 

    Entradas: 
        - dir (str): path absoluto a la carpeta donde se encuentran los archivos '.h5' que se quieren suavizar
        - n_keypoints (int): numero de keypoints trackeados con DeepLabCut
        - umbral_likelihood (float): valor minimo aceptado de likelihood para un determinado trackeo. Posiciones con likelihoods menores a este valor seran 
                                        interpoladas. Default: 0.99
    
    Salidas: 
        Guarda en la carpeta de trabajo copias de los archivos h5 pero suavizados. Agrega '_clean' al nombre para no sobre-escribir los originales.
    """

    os.chdir(dir)
    VIDEOfiles_names = sorted(glob.glob('*.h5'))
    for vid in VIDEOfiles_names:

        track = pd.read_hdf(vid)

        likelihoods = np.zeros(shape=(len(track), n_keypoints))

        for i in range(n_keypoints):

            likelihoods[:,i] = (track[track.columns.get_level_values(0)[0],track.columns.get_level_values(1).unique()[i], 'likelihood'])

        likelihoods = pd.DataFrame(likelihoods, columns=track.columns.get_level_values(1).unique())

        for index,row in likelihoods.iterrows():

            if index > 0:
                outlier = any(row < umbral_likelihood)
                if outlier:
                    track.iloc[index, :] =[np.nan for x in track.iloc[index,:]]


        track = track.interpolate()
        name = vid[:-3] + '_clean.h5'
        track.to_hdf(name, key='track')

def get_keypoints_pos(track:pd.DataFrame):
    """
    Esta funcion se encarga de extraer las posiciones en x y en y de cada keypoint a partir del dataframe generado por DLC
    Entradas:
        - track: DataFrame salida de DLC (crudo o pre-procesado por la funcion clean)
    Salidas:
        - xpositions: DataFrame con las posiciones en x de cada keypoint
        - ypositions: DataFrame con las posiciones en y de cada keypoint
    
    """ 
    bodyparts = np.array([x for x in track.columns.get_level_values(1)]) #extraemos las partes del cuerpo (keypoint)
    n_keypoints = len(np.unique(bodyparts)) #guardamos el numero de keypoints
    #inicializamos los DF salida
    xpositions = pd.DataFrame(np.zeros((len(track), n_keypoints)), columns=np.unique(bodyparts)) 
    ypositions = pd.DataFrame(np.zeros((len(track),n_keypoints)), columns=np.unique(bodyparts))

    ##extraemos las posiciones en x y en y
    for i in range(n_keypoints):
        xpositions[np.unique(bodyparts)[i]] = (track[track.columns.get_level_values(0)[0], np.unique(bodyparts)[i], 'x'])

    ypositions = pd.DataFrame(np.zeros((len(track),n_keypoints)), columns=np.unique(bodyparts))
    for i in range(n_keypoints):
        ypositions[np.unique(bodyparts)[i]] = (track[track.columns.get_level_values(0)[0], np.unique(bodyparts)[i], 'y'])
    
    return xpositions, ypositions

def get_closest_keypoint(xpos: pd.DataFrame, ypos:pd.DataFrame, objCoords:list, pix_to_cm: float):
    """
    Esta funcion se encarga de encontrar el keypoint mas cercano a un objeto o coordenadas del espacio para cada frame de video

    Args:
        xpos (pd.DataFrame): contiene las posiciones en x para cada keypoint 
        ypos (pd.DataFrame): contiene las posiciones en y para cada keypoint
        objCoords (list): coordenadas de referencia 
        pix_to_cm (float): ratio para medir en cm 

    Returns:
        min_indices_list (list): contiene el numero de keypoint mas cercano para cada frame
    """
     # Convert the DataFrames to NumPy arrays for vectorized operations
    xpos_np = xpos.to_numpy()
    ypos_np = ypos.to_numpy()
        # Calculate the positions in cm
    positions_x = np.round(xpos_np / pix_to_cm)
    positions_y = np.round(ypos_np / pix_to_cm)
        # Combine the positions into a single array of shape (num_rows, num_cols, 2)
    positions = np.stack((positions_x, positions_y), axis=-1)
            # Calculate the Euclidean distances between each position and the object coordinates
    obj_coords_array = np.round(np.array(objCoords))
        # Initialize an array to store the minimum distances
    min_indices_list = []

    for pos in positions:
            # Expand the dimensions of pos to make it (num_keypoints, 1, 2)
            pos_expanded = np.expand_dims(pos, axis=1)
            
            # Calculate the Euclidean distances between each keypoint and each object
            distances = np.linalg.norm(pos_expanded - obj_coords_array, axis=-1)
            
            # Find the minimum distance for each keypoint across all objects
            min_distances = np.min(distances, axis=1)
            
            # Find the index of the keypoint with the minimum distance
            min_index = np.argmin(min_distances)
            
            min_indices_list.append(min_index)
    return min_indices_list

def get_position_closest(xpos:pd.DataFrame, ypos: pd.DataFrame, closest_keypoint: list, pix_to_cm: float):
    """
        Esta funcion genera un dataframe de posiciones en x e y para cada frame utilizando el keypoint mas cercano al objeto o punto de referencia (obtenido con get_closest_keypoint)


    Args:
        xpos (pd.DataFrame): contiene las posiciones en x para cada keypoint 
        ypos (pd.DataFrame): contiene las posiciones en y para cada keypoint
        closest_keypoint (list): lista con el numero de keypoint mas cercano en cada frame
        pix_to_cm (float): ratio para medir en cm 
    Returns:
        pos (pd.DataFrame): contiene dos columnas 'posx' y 'posy' que tienen las posiciones del keypoint mas cercano por frame (fila)
    """
    pos = pd.DataFrame(np.zeros(shape=(len(closest_keypoint),2)), columns=['posx', 'posy'])
    for j,keypoint in enumerate(closest_keypoint):
        pos.iloc[j, 0] = round(xpos.iloc[j,keypoint]/pix_to_cm)
        pos.iloc[j, 1] = round(ypos.iloc[j,keypoint]/pix_to_cm)
    return pos


def get_centroids(xpositions: pd.DataFrame, ypositions:pd.DataFrame):
    """
    Esta funcion obtiene la posicion del centroide del pez para cada frame. El centroide se aproxima como la mediana de las posiciones en x y en y de todos los 
    keypoints trackeados. 
    Entradas: 
        - track (pd.DataFrame): DataFrame con el trackeo obtenido por DLC para un video (informacion guardada en un archivo .h5). 
        - n_keypoints (int): numero de keypoints trackeados
    Salidas: 
        - centroids (list): lista de listas donde cada elemento almacena los pares (x,y) para una frame. 
    """
    centroids = []
    median_xposition = np.nanmedian(xpositions, axis=1) #xpos
    median_yposition = np.nanmedian(ypositions, axis=1) #ypos

    for j in range(len(median_xposition)):
        centroids.append([median_xposition[j], median_yposition[j]])
    
    return centroids

def calculate_velocity(x: list, sf:float, pix_to_cm:float):
    """
    Esta funcion calcula la velocidad de una serie de posiciones consecutivas. Calcula la distancia entre dos pares (x,y) consecutivos y lo divide por el tiempo
    entre frames consecutivas.
    Entradas: 
            - x (list): lista que contenga los pares (x,y) para cada frame. Tambien funciona con una np.array de tamano (n_frames x 2)
            - sf (float): frecuencia de muestreo del trackeo (frames por segundo)
            - pix_to_cm (float): relacion entre los pixeles de la imagen y la distancia real en cm. Ajustar para calibrar al experimento particular.

    Salida:
            - v (list): lista que contiene las velocidades instantaneas para cada frame presentada en x.
    """
    desplazamiento = [abs(distance.euclidean(x,y)/pix_to_cm) for x, y in zip(x[1:], x[:-1])]
    dt = 1 / sf
    v = [i/dt for i in desplazamiento]
    return v

def calc_time_moving(files: list, v_umbral: float):
    """
    Esta funcion calcula el porcentaje total de tiempo que el animal pasa moviendose a velocidades mayores a un determinado umbral
    Entradas: 
            - files (list): lista con los nombres de los archivos que se quiere utilizar para calcular el tiempo en movimiento (deben estar en la misma carpeta de trabajo)
            - v_umbral (float): velocidad umbral para determinar que se considera movimiento. Unidades: cm/s
    Salida: 
            - time_moving (float) : porcentaje del tiempo total de los archivos brindados donde la velocidad es mayor al umbral. 
    """
    movement = 0
    total = 0 
    for file in files:
            track = pd.read_hdf(file)
            xpos, ypos = get_keypoints_pos(track)
            centroids = get_centroids(xpos, ypos)
            velocity = calculate_velocity(centroids, sf=50, pix_to_cm=10)

            idx_movement = [i for i,x in enumerate(velocity) if x > v_umbral] #filtramos frames con mas de umbral de v

            movement += len(idx_movement)
            
            total += len(track)

    time_moving = (movement * 100 / total)
    return time_moving

def get_visit_density_grid(files_h5: list, pix_to_cm:float, x_dim: int, y_dim:int, dis_centroids = True, objCoords=None):
    """
    Esta funcion se encarga de generar la matriz con el tiempo de visita por cm cuadrado de pecera. 

    Args:
        files_h5 (list): lista con los paths a los archivos h5 de DLC (o los pre-procesados)
        pix_to_cm (float): ratio para medir en cm 
        x_dim (int): ancho en cm de la pecera
        y_dim (int): alto en cm de la pecera
        dis_centroids (bool, optional): Boolean para decidir si usar centroides o keypoint mas cercano como criterio para la ubicacion del animal. Default es true (usar centroides)
        objCoords (_type_, optional): En caso de usar el keypoint mas cercano, proveer coordenadas de referencia. Default es None.

    Returns:
        time_grid (np.array): grid con los valores de tiempo de visita para cada cm2 de pecera
         
    """
    
    time_grid =  np.zeros(shape=(x_dim,y_dim))
    Total = 0
    #objCoords = [round(x/pix_to_cm) for x in objCoords]        
    closest_keypoints_all = []
    for file in files_h5:
        vid = pd.read_hdf(file)   
        xpos, ypos = get_keypoints_pos(vid)   
        if dis_centroids:
            pos = get_centroids(xpos, ypos)
        else:
            closest_keypoint = get_closest_keypoint(xpos, ypos, objCoords, pix_to_cm)
            pos = get_position_closest(xpos, ypos, closest_keypoint, pix_to_cm)
            closest_keypoints_all = closest_keypoints_all + closest_keypoint
        
        grouped_coords = pos.groupby(['posx', 'posy'])
        timeDensity_count =  grouped_coords.size()
        for coord, count in timeDensity_count.items():
            time_grid[int(coord[0]),int(coord[1])] += count
            Total += count 
    
    time_grid[time_grid == 0 ] = np.nan
    time_grid = time_grid*100/(Total)
    print('Grid de tiempo completa, suma de elementos:  ' + str(np.nansum(time_grid)))
    return time_grid

def get_vid_EOD_time(file_h5: str, file_bin:str, sf: float, date: str, n_channels : int):
    """
    Esta funcion crea el vector de tiempo para la EOD y para el video

    Args:
        file_h5 (str):  lista con los paths a los archivos h5 de DLC (o los pre-procesados)
        file_bin (str):  lista con los paths a los archivos bin de registro electrico
        sf (float): frecuencia de muestreo para el registro electrico
        date (str): dia de registro segun el formato 'AAAA-MM-DD'
        n_channels (int): numero de canales de registro

    Returns:
        time_EOD (np.array): vector de tiempo del registro electrico en segundos
        videoTime (np.array): vector de tiempo del video en segundos
    """
    files_start = datetime.strptime(date, '%Y-%m-%dT%H_%M_%S')
    midnight = files_start.replace(hour=0, minute=0, second=0, microsecond=0) #definimos la media noche para el dia donde se registro ese archivo
    start = abs(midnight - files_start).total_seconds() # calculamos el tiempo de inicio del archivo en segundos totales respecto de las 00 para poder compararla
    
    
    EOD = np.fromfile(file_bin,dtype=np.int32)

    EOD_ch = EOD_ch = EOD.reshape((int(EOD.shape[0]/n_channels),n_channels))
    time_EOD = np.linspace(start=start, stop=start+EOD_ch.shape[0]/sf, num=EOD_ch.shape[0])
    del EOD

    vid = pd.read_hdf(file_h5)
    durationVideo = len(vid)/sf 
    videoTime = np.linspace(start=start, stop=start+durationVideo, num=len(vid))
    
    return time_EOD, videoTime

def correlate_vid_EOD(file_h5: list, file_bin:list, pkl_file: dict, sf: float, pix_to_cm: float, date: str, n_channels:int,  dis_centroids = True, objCoords=None):
    """
    Esta funcion genera un dataframe que contiene los datos de video y de FB-DOE. Usa los vectores de tiempo del video y de la EOD para asociar la frecuencia basal para cada frame de video.

    Args:
        file_h5 (str):  lista con los paths a los archivos h5 de DLC (o los pre-procesados)
        file_bin (str):  lista con los paths a los archivos bin de registro electrico
        pkl_file (dict): diccionario generado por el analisis de la DOE que tiene los valores de FB-DOE para todos los videos
        sf (float): frecuencia de muestreo del registro electrico
        pix_to_cm (float): ratio para medir en cm
        date (str):  dia de registro segun el formato 'AAAA-MM-DD'
        n_channels (int): numero de canales de registro
        dis_centroids (bool, optional): Boolean para decidir si usar centroides o keypoint mas cercano como criterio para la ubicacion del animal. Default es true (usar centroides)
        objCoords (_type_, optional): En caso de usar el keypoint mas cercano, proveer coordenadas de referencia. Default es None.


    Returns:
        pos (pd.DataFrame): contiene las posiciones en x y en y (del centroide o del keypoint mas cercano) y la frecuencia basal para cada frame
    """
    
    freq = np.array(pkl_file['FB-DOE'][date]) 
    time_EOD, videoTime = get_vid_EOD_time(file_h5, file_bin, sf, date=date, n_channels=n_channels)
    f_video = []
    freqT = np.take(time_EOD, [x for x in pkl_file['Peak-time'][date] if x <len(time_EOD)])
    for i in range(len(videoTime)):
        dif = abs(freqT - videoTime[i])
        closest = np.min(dif)
        index = dif.tolist().index(closest)
        fi = freq[index]
        f_video.append(fi)
        
    vid = pd.read_hdf(file_h5)

    xpos, ypos = get_keypoints_pos(vid) 
    if dis_centroids:
        pos = get_centroids(xpos, ypos)
    else:
        closest_keypoint = get_closest_keypoint(xpos, ypos, objCoords, pix_to_cm)
        pos = get_position_closest(xpos, ypos, closest_keypoint, pix_to_cm)

    pos['f_vid'] = f_video
        
    return pos        
    
def get_frequency_grid(files_h5: list, files_bin: list, pkl_file: dict, sf: float, pix_to_cm:float, x_dim: int, y_dim:int, n_channels:int,  dis_centroids = True, objCoords=None):
    """
    Esta funcion genera la grid de frecuencia media por cm2. 

    Args:
        files_h5 (str):  lista con los paths a los archivos h5 de DLC (o los pre-procesados)
        files_bin (str):  lista con los paths a los archivos bin de registro electrico
        pkl_file (dict): diccionario generado por el analisis de la DOE que tiene los valores de FB-DOE para todos los videos
        sf (float): frecuencia de muestreo del registro electrico
        pix_to_cm (float): ratio para medir en cm
        x_dim (int): ancho en cm de la pecera
        y_dim (int): alto en cm de la pecera
        n_channels (int): numero de canales de registro
        dis_centroids (bool, optional): Boolean para decidir si usar centroides o keypoint mas cercano como criterio para la ubicacion del animal. Default es true (usar centroides)
        objCoords (_type_, optional): En caso de usar el keypoint mas cercano, proveer coordenadas de referencia. Default es None.


    Returns:
        freq_grid: grid con los valores de FB-DOE media para cada cm2 de paecera
    """
    
    freq_grid = np.zeros(shape=(x_dim, y_dim), dtype=float)
    coordsf_list = []
    
    for i, date in enumerate(sorted(pkl_file['FB-DOE'].keys())):
        coordsf = correlate_vid_EOD(files_h5[i], files_bin[i], pkl_file, sf, pix_to_cm, date=date, n_channels=n_channels, dis_centroids=dis_centroids, objCoords=objCoords)
        coordsf_list.append(coordsf)
    
    coordsf = pd.concat(coordsf_list)
    
    grouped_coords = coordsf.groupby(['posx', 'posy']).median().reset_index()
    
    for i in range(len(grouped_coords)):
        freq_grid[int(grouped_coords.iloc[i, 0]), int(grouped_coords.iloc[i, 1])] = grouped_coords.iloc[i, 2]
    
    freq_grid[freq_grid == 0] = np.nan
    return freq_grid

def mean_grid_by_distance(grid: np.array,  obj_coordinates: list, pix_to_cm:float, limit: int):
    """
    Esta funcion toma una grid y un punto de referencia y calcula la media de la variable en funcion a la distancia del punto de referencia

    Args:
        grid (np.array): grid espacial de una de las variables (tiempo, FB-DOE o densidad de muestreo)
        obj_coordinates (list): coordenadas del punto de referencia
        pix_to_cm (float): ratio para medir en cm
        limit (int): cuantos cm de distancia usar para calcular

    Returns:
        mean_variable_distance (np.array): contiene la distancia en cm y el valor para cada distancia
    """

    grid_by_distance = np.zeros(shape=(grid.shape[0]*grid.shape[1], 4))
    for x, fila in enumerate(grid):
        for y, valor in enumerate(fila):
            grid_by_distance[y+len(fila)*x, 0] = x
            grid_by_distance[y+len(fila)*x, 1] = y
            grid_by_distance[y+len(fila)*x, 2] = valor
    
    i=0
    for x, y in zip(grid_by_distance[:,0], grid_by_distance[:,1]):
        dis = []
        for obj in obj_coordinates:
            dis.append(round(distance.euclidean([x,y], [j/pix_to_cm for j in obj])))
        grid_by_distance[i, 3] = np.min(dis)
        i+=1

   
    grid_by_distance = pd.DataFrame(grid_by_distance[:,2:], columns=['variable', 'distance'])
    grid_by_distance_grouped = grid_by_distance.groupby('distance')
    
    mean_variable_distance = np.zeros(shape=(limit, 2))
   
    i=0
    for name, group in grid_by_distance_grouped:
        if int(name) in range(1,limit):
            mean_variable_distance[i, 0] = int(name)
            mean_variable_distance[i,1] = np.sum(group['variable'])
            i+=1
        
    return mean_variable_distance

def plot_map(grid: np.array, objCoordinates: list, label: str, pix_to_cm:float, filename:str , vmax=None,vmin=None, cmap='viridis'):
    """
    Esta funcion genera los mapas de distribucion espacial de la variable que se le de

    Args:
        grid (np.array): grid a graficar
        objCoordinates (list): puntos de referencia a graficar, si se deja vacio no se grafica nada
        pix_to_cm (float): ratio para medir en cm
        label (str): label de la variable para la barra de color
        filename (str): nombre con cual guardar la figura
        vmax (float, optional): valor maximo para el rango de colores. Defaults to None.
        vmin (float, optional): valor minimo para el rango de colores. Defaults to None.
        cmap (str, optional): colormap a utilizar para el heatmap. si no se indica se usa viridis
        
    """
    fig, ax = plt.subplots()
    plt.imshow(grid, cmap=cmap, vmax=vmax, vmin=vmin, origin='lower')
    cbar = plt.colorbar()
    cbar.set_label(label)
    if objCoordinates != None:
        x = objCoordinates[0]/pix_to_cm
        y =  objCoordinates[1]/pix_to_cm
        plt.scatter(x, y, s=100, c='r', marker='*')

    
    plt.show()
    fig.savefig(filename, format='svg', dpi=1200)