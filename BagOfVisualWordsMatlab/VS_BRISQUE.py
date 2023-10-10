import cv2 as cv
from PIL import Image
import os
import numpy as np
import random
from matplotlib import pyplot as plt
from brisque import BRISQUE
import matplotlib.cm as cm
import scipy.misc
import scipy.ndimage as ndi
from skimage import filters
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from PIL import Image, ImageOps




dir_imagenes = 'C:/Users/Knowd/.spyder-py3/asl_alphabet_train/asl_alphabet_train'

# Inicializa el objeto BRISQUE
obj = BRISQUE()

# Número de imágenes a procesar
num_images_to_process = 50

# Obtiene una lista de rutas a las imágenes en el directorio
lista_imagenes = [os.path.join(dirpath, filename)
                  for dirpath, dirnames, filenames in os.walk(dir_imagenes)
                  for filename in filenames if filename.endswith('.jpg')]

# Selecciona 50 imágenes al azar
selected_images = random.sample(lista_imagenes, num_images_to_process)

# Listas para almacenar puntajes BRISQUE
scores_originales = []
scores_procesados = []

# Parametros de escalado
new_width = 1000
new_height = 1000

# Itera a través de las imágenes seleccionadas
for ruta_imagen in selected_images:
    # Carga la imagen utilizando OpenCV
    imagen_original = cv.imread(ruta_imagen)
    
    #Normalizar la intensidad de pixeles
    imagen_original = cv.normalize(imagen_original, None, 0, 255, cv.NORM_MINMAX)
    

    

    
    # Aplica los filtros a la imagen original
    resized_image_original = cv.resize(imagen_original, (new_width, new_height), interpolation=cv.INTER_NEAREST)
    
    # Obtiene las dimensiones de la imagen original
    altura, ancho, canales = resized_image_original.shape
    
    
    
    
    
    # Filtros para la imagen original
    kernel_laplacian = np.array([[-1, -1, -1],
                                 [-1, 9, -1],
                                 [-1, -1, -1]])
    kernel_boxBlur = np.array([[1/9, 1/9, 1/9],
                               [1/9, 1/9, 1/9],
                               [1/9, 1/9, 1/9]])
    

    brightness = 1.5
    contrast = 1.4
    
    # Calcular el puntaje BRISQUE de la imagen original
    score_original = obj.score(imagen_original)
    
    img_gaussian_laplacian = cv.filter2D(resized_image_original, -1, kernel_laplacian)
    img_gaussian_laplacian = cv.GaussianBlur(img_gaussian_laplacian, (3, 3), 0)
    imagen_ajustada = cv.convertScaleAbs(img_gaussian_laplacian, alpha=contrast, beta=brightness)
    
    
    
    
    
    
    # Calcular el puntaje BRISQUE de la imagen procesada
    score_procesada = obj.score(imagen_ajustada)
    
    # Muestra ambas imágenes junto con sus puntajes
    plt.figure(figsize=(12, 6))
    
    # Muestra la imagen original procesada
    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(imagen_original, cv.COLOR_BGR2RGB))
    plt.title(f"Original \nBRISQUE Score: {score_original:.2f}")
    plt.axis('off')
    
    # Muestra la imagen procesada
    plt.subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(imagen_ajustada, cv.COLOR_BGR2RGB))
    plt.title(f"Procesada\nBRISQUE Score: {score_procesada:.2f}")
    plt.axis('off')

    # Muestra la figura actual
    plt.show()
    
    # Calcula el puntaje BRISQUE de la imagen original
    score_original = obj.score(imagen_original)
    scores_originales.append(score_original)

    # Calcula el puntaje BRISQUE de la imagen procesada
    score_procesada = obj.score(imagen_ajustada)
    scores_procesados.append(score_procesada)
    
# Calcula el promedio de los puntajes BRISQUE
promedio_original = np.mean(scores_originales)
promedio_procesado = np.mean(scores_procesados)

print(f'Promedio BRISQUE de imágenes originales: {promedio_original:.2f}')
print(f'Promedio BRISQUE de imágenes procesadas: {promedio_procesado:.2f}')

# Encuentra el índice de la imagen con el puntaje BRISQUE más alto entre las imágenes procesadas
indice_max_score_procesado = np.argmax(scores_procesados)

# Encuentra el índice de la imagen con el puntaje BRISQUE más bajo entre las imágenes procesadas
indice_min_score_procesado = np.argmin(scores_procesados)

# Encuentra el índice de la imagen cuyo puntaje BRISQUE está más cerca del promedio original entre las imágenes procesadas
indice_cercano_promedio_procesado = np.argmin(np.abs(np.array(scores_procesados) - promedio_procesado))

# Muestra la imagen con el puntaje BRISQUE más alto entre las imágenes procesadas
imagen_max_score_procesado = cv.imread(selected_images[indice_max_score_procesado])
plt.imshow(cv.cvtColor(imagen_max_score_procesado, cv.COLOR_BGR2RGB))
plt.title(f"Imagen con el puntaje BRISQUE más alto entre las procesadas: {scores_procesados[indice_max_score_procesado]:.2f}")
plt.axis('off')
plt.show()

# Muestra la imagen con el puntaje BRISQUE más bajo entre las imágenes procesadas
imagen_min_score_procesado = cv.imread(selected_images[indice_min_score_procesado])
plt.imshow(cv.cvtColor(imagen_min_score_procesado, cv.COLOR_BGR2RGB))
plt.title(f"Imagen con el puntaje BRISQUE más bajo entre las procesadas: {scores_procesados[indice_min_score_procesado]:.2f}")
plt.axis('off')
plt.show()

# Muestra la imagen con el puntaje BRISQUE más cercano al promedio original entre las imágenes procesadas
imagen_cercano_promedio_procesado = cv.imread(selected_images[indice_cercano_promedio_procesado])
plt.imshow(cv.cvtColor(imagen_cercano_promedio_procesado, cv.COLOR_BGR2RGB))
plt.title(f"Imagen más cercana al promedio BRISQUE entre las procesadas: {scores_procesados[indice_cercano_promedio_procesado]:.2f}")
plt.axis('off')
plt.show()