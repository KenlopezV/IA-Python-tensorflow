import numpy as np
import tensorflow as tf
from tf.keras.utils import load_img, img_to_array
from keras.models import load_model

longitud, altura = 150, 150 #tiene que ser la misma que se definio en el entrenamiento
modelo = './modelo/modelo.h5' #ruta de directorio
pesos_modelo = './modelo/pesos.h5'
cnn = load_model(modelo) #a nuestra variable cnn le cargamos el modelo que tenemos en el directorio
cnn.load_weights(pesos_modelo)

def predict(file):
  x = load_img(file, target_size=(longitud, altura)) #cargamos una imagen a predecir en nuestra variable x
  x = img_to_array(x) #para convertir nuestra imagen en un arreglo
  x = np.expand_dims(x, axis=0) #añadimos una dimecion extra en el eje 0 para procesar nuestra informacion sin problema
  array = cnn.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    print("pred: La imagen es de un perro")
  elif answer == 1:
    print("pred: La imagen es de un gato")
  elif answer == 2:
    print("pred: La iamgen es de un gorila")

  return answer

predict('descarga.jpg')  