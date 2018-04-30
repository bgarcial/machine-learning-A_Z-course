# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer

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

# Taking care of missing data
# Tomaremos la media de las columnas, reemplazaremos los datos
# que faltan aqui por la media de todos los valores en la columna Age
# de este dataset Data.csv
# Lo haremos con scikit-learn con Imputer que es uno
# de los diferentes clases para preporcesamiento de datos de
# scikit learn
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html
# strategy es lo que haremos para llenar los daots faltantes, en este caso
# la media de los otros valores de esa columna
# axis es el eje a lo largo del cual hacer el impute, si es = 0 significa que
# sera a lo largo de las columnas
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

# Fijamos los valores transformando las columnas en donde hace falta datos
# con fit le pasamos el dataset x y todas las lineas de la columna con los
# primeros : y luego seleccionamos la columna de la derecha que es la primera 1
# a la que le falta datos y hasta la columna 3 que nos llega a la de salaario,
# esto es porque 0-pais 1-Edad 2-Salario 3-Purchased, pero ponemos 3 porque en
# python, los slices, el limite superior es excluido, al poner 3 nos da la columna
# 2 que es Salario que es en donde hay otro dato faltante y de esta manera estamos
# tomando los indices 0,1 y 2 con imputer
imputer = imputer.fit(X[:, 1:3])

# Y ahora reemplazamos los datos faltantes en el dataset X descritos antes
# por la media de los demas valores de dichas columnas en donde faltan datos
# Seleccionamos las columnas en donde faltan datos
# tomamos todas los valores (:) de las columnas cuyos indices son 1 y 3
# y usamos el metodo transform el cual reemplazara los datos faltantes
# por el promedio de la columna y le pasamos las columnas sobre las cuales actuar.
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
"""
En Data.csv tenemos dos variables categóricas
Country y Purchased son variables categoricas porque simplemente ellas contienen categorias 
Country tiene tres categorias : France, Spain, Germany
Purchased tiene dos categorias: Yes, No.

Dado que los modelos de aprendizaje automático se basan en 
ecuaciones matemáticas, puedes entender intuitivamente que 
causaría algún problema si mantenemos el texto de estas variables
categóricas en las ecuaciones (los paises y Yes, No.)
Solo queremos numeros en las ecuaciones asi que necesitamos codificar las variables categóricas. 
Eso es codificar el texto que tenemos aquí en números
Entonces codificaremos las dos variables de country y purchased
"""


# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Creamos un objeto LabelEncoder()
labelencoder_X = LabelEncoder()

# Del dataset X tomamos la columna 0-Country
# y usamos el metodo fit_transform solo sobre la primera columna de X
# es decir Country, tomando todas sus lineas o rows :
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

"""
Si miramos vemos que la salida contiene 10 valores
de la primera columna de nuestra matriz X

X[:,0]
array([0, 2, 1, 2, 1, 0, 2, 0, 1, 0], dtype=object)
Aqui se han codificado los valores de la columna Country
X
array([['France', 44.0, 72000.0],
       ['Spain', 27.0, 48000.0],
       ['Germany', 30.0, 54000.0],
       ['Spain', 38.0, 61000.0],
       ['Germany', 40.0, 63777.77777777778],
       ['France', 35.0, 58000.0],
       ['Spain', 38.77777777777778, 52000.0],
       ['France', 48.0, 79000.0],
       ['Germany', 50.0, 83000.0],
       ['France', 37.0, 67000.0]], dtype=object)
X
array([[0, 44.0, 72000.0],
       [2, 27.0, 48000.0],
       [1, 30.0, 54000.0],
       [2, 38.0, 61000.0],
       [1, 40.0, 63777.77777777778],
       [0, 35.0, 58000.0],
       [2, 38.77777777777778, 52000.0],
       [0, 48.0, 79000.0],
       [1, 50.0, 83000.0],
       [0, 37.0, 67000.0]], dtype=object)

Entonces aqui hemos codificado la columna del pais
Sin embargo, ¿sospechas que algún problema podría suceder?
El problema es que los modelos de aprendizaje se basan en 
ecuaciones y eso es bueno que y es bueno que hayamos 
reemplazado el texto por números para que podamos incluir 
los números en las ecuaciones
Sin embargo, dado que uno es mayor que cero y dos es mayor que uno 
las ecuaciones en el modelo pensaran que España tiene un valor mas 
alto que Alemania y Francia

Y que Alemania tiene un valor mas alto que Francia

Y este no es el caso
Estas son en realidad tres categorías y no hay un orden 
relacional entre las tres. NO se puedne comparar
diciendo que Spain es mas grnade qie Germany o que 
GErmany es mas grande que France, esto no tiene  ningun sentido

Si tuviéramos, por ejemplo, el tamaño variable con el tamaño como 
pequeño, mediano y grande, entonces sí podríamos expresar órdenes 
entre los valores de esta variable porque grande es mayor que el mediano 
y el mediano es mayor que el pequeño 

Asi que para prevenir que las ecuaciones del aprendizaje automatico piensen 
que Germany es mas grande que France y SPain mas grande que Germany usaremos una variable
llamada dummy variables     
"""
# https://docs.google.com/document/d/1akyF1QxKp_QvBzv8m2hYUe5R5SX_jIK55Ro7dNSTj9M/edit#bookmark=id.bv9hskqqnqh1
# Creamos un objeto OneHotEncoder con el atributo categorical_features que
# especifica que features son tratadas como categoricas, que en este caso el valor sera
# el arreglo de los elementos de la columna 0, que es Country
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
onehotencoder = OneHotEncoder(categorical_features = [0])

# Aplicamos lo anterior a nuestra matriz o dataset X
# usando el metodo fit_transform
# Y solo tenemos que tomar la primera columna de X porque se
# le especificó antes a índice cero
X = onehotencoder.fit_transform(X).toarray()
"""
X
array([[1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.40000000e+01,
        7.20000000e+04],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 2.70000000e+01,
        4.80000000e+04],
       [0.00000000e+00, 1.00000000e+00, 0.00000000e+00, 3.00000000e+01,
        5.40000000e+04],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 3.80000000e+01,
        6.10000000e+04],
       [0.00000000e+00, 1.00000000e+00, 0.00000000e+00, 4.00000000e+01,
        6.37777778e+04],
       [1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.50000000e+01,
        5.80000000e+04],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 3.87777778e+01,
        5.20000000e+04],
       [1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.80000000e+01,
        7.90000000e+04],
       [0.00000000e+00, 1.00000000e+00, 0.00000000e+00, 5.00000000e+01,
        8.30000000e+04],
       [1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.70000000e+01,
        6.70000000e+04]])
Como resultado miramos el dataset resultante
https://docs.google.com/document/d/1akyF1QxKp_QvBzv8m2hYUe5R5SX_jIK55Ro7dNSTj9M/edit#bookmark=id.ozkglnqoueuf

Y luego establecemos una comparación ...
https://docs.google.com/document/d/1akyF1QxKp_QvBzv8m2hYUe5R5SX_jIK55Ro7dNSTj9M/edit#bookmark=id.sah1h6o72o49
Y observemos los resultados basados en esta codificacion de la variable
categorica Country en relacioncon las nuevas columnas credas a partir
de sus categorias
https://docs.google.com/document/d/1akyF1QxKp_QvBzv8m2hYUe5R5SX_jIK55Ro7dNSTj9M/edit#bookmark=id.g8zs89wwujme
y el dataset original 


"""

# Encoding the Dependent Variable
"""
Y ahora alégrese de que no tendremos que usar el codificador 100, 
solo necesitaremos utilizar el codificador de etiquetas, ya que como 
la variable dependiente, el modelo de aprendizaje automático sabrá que 
es una categoría y que no hay un orden entre los dos.
"""

"""
 Tomamos el vector de variables dependientes que es solo
la ultima columna purchase, es decir la tercera y usamos 
el label encoder para crear un objeto para Y, dado que el anterior ya
fue transformado para X 
"""
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

"""
Es asi de esta manera como hemos creado nuestros LabelEncodeer objects 
que ajustaran a X e Y y los transformaran a X en una matriz independient
y a Y en un vector de variable dependiente 
"""