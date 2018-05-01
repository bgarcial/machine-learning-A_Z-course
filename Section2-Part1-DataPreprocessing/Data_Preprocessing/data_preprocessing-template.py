"""
Importamos las librerias
Importamos los datasets
Llenamos datos faltantes
Codifiamos variables categoricas
Dividios el dtaset en entreamiento y pruebas
Aplicamos escalamiento de caracteristicas para poner
todsa las variables en la misma escala

Sin embargo aqui no pondremos todo junto

Solo importaremos las librerias
Con respecto a los datos que faltan, solo quería mostrarte cómo encargarme de eso.
En caso de que tengas algunos datos faltantes en tu conjunto de datos que conoces en tu trabajo,
no sera incluido tampoco

ES solo bueno saber como abordar esto pero nos enfocaremos en modelos de ML
La Codificacion de variables categoricas tampoco sera incluida
porque vamos a encontrar muy pocos ejemplos de datos donde tenemos que codificar los datos

Principalmente nos enfocaremos en modelos de ML

Si incluimos la division en dataset de entrenamiento y data set de pruebas pues este es un paso muy importante,
necesitamos dividirlo entre training y test porque enesitamos evaliar nuestro modelo con diferentes
conjuntos de datos


"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('../../data/Data.csv')

"""
En un dataset necesitamos distinguir la matriz de características
y el vector de la variable dependiente
"""
# Creamos la matriz de tres variables independientes
# Las columnas Country, Age, Salary son independientes
# Tomamos todas las columnas - la de Purchase que es
# dependiente
X = dataset.iloc[:, :-1].values

# Creamos el vector de variables dependientes que es solo
# la ultima columna purchase, es decir la tercer
y = dataset.iloc[:, 3].values



# Splitting the dataset into the Training set and Test set
#Importaremos la libreria de cross validation
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""


"""
Si copiamos esto y lo colocamos al comienzo de nuestros modelos de ML
antes de construir el modelo en si mismo .
Esto nos ahorrara mucho tiempo y no tendremos que cambiar muchas cosas
 otras si como, por ejemplo, a veces cambiaremos el test_size del modelo de entrenamiento y de pruebas
 o los indices cuando importamos el dataset y destacamos als entradas y salidas
 puede que tengamos mas de tres variables independientes o menos
 y los indices en el caso del -1 es que remueve la ultima columna 
 
 El nombre de nuestro dataset tambien lo cambiaremos
 EN el pero de los casos, solo necesitaremos cambiar tres cosas
 
- El nombre del dataset
- El indice de la columna de la variable dependiente
- el test_size 

"""