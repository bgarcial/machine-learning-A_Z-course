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

# --------------------------------
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
Y ahora alégrese de que no tendremos que usar el codificador para purchased, 
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
y = labelencoder_X.fit_transform(y)

"""
Es asi de esta manera como hemos creado nuestros LabelEncodeer objects 
que ajustaran a X e Y y los transformaran a X en una matriz independient
y a Y en un vector de variable dependiente 
"""





# Splitting the dataset into the Training set and Test set
#Importaremos la libreria de cross validation
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

"""
# Construimos nuestro dataset de pruebas
Lo que haremos es crear las variables 
X_train, X_test y_train, y_test

X_train es la parte de entrenamiento de la matriz de 
caracteristicas

X_test es la parte de pruebas de la matriz de caracteristicas

y_train es la parte de entrenamiento de las variables dependientes que es
asociada a X_train

Esto significa que tenemos los mismos índices para ámbos con las mismas observaciones

y_test  es la parte de pruebas del vector de la variable dependiente asociado a X_test
y entonces y definiremos sus valores al mismo tiempo. 

Parámetros
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

El primer parametro debe ser un array, asi que le pasamos la Variable X 
categorizada que contiene la primera columna de X es decir Country 
esta es la matriz X de variables independientes

y es el vector de variable dependiente

Con X e y estamos colocando el dataset entero, ambos son arreglos

El proximo parametro test_size y es el tamaño del dataset de pruebas que queremos
escoger, asi por ejemplo si colocamos  test_size = 0.5
eso es 50 por ciento que significa que va a haber la mitad de sus datos yendo al 
conjunto de prueba y la mitad de sus datos van a un conjunto de entrenamiento

Una buena elección para el tamaño de pruebas es generalmente 0.2 es decir el 20%  o 0.25 
o incluso 3 porciento
EN algunos casos raros tendremos el 4% pero casi nunca el 0.5

Escogemos el 2% lo que significa que tendremos 10 muestras 
Entonces eso significa que, dado que tenemos 10 observaciones, es bueno regresar 
a la exploración anulable y encontrar otra. 

Tenemos 10 observaciones. Entonces eso significa que una vez que hagamos la prueba 
entrenada, tendremos dos observaciones en el conjunto de prueba y ocho observaciones 
en el conjunto de entrenamiento.

Y es todo lo que tenemos para las entradas
Miramos que el proximo parametro es  train_size, pero 
Pero como test_size plus train_size es igual a uno, no es necesario colocarlo ya 
que sería redundante

Entonces usaremos random_state para generar resultados iguales a los del curso
solo para este caso del curso lo usaremos.
Entonces con el random_state establecemos a 0 
para tener los mismso resultados que en el demo

Y entonces, si miramos tenemos 8 muestras en X_train y 2 en X_test

y LO MISmo para la salida, en y_train tenemos 8 muestras y en y_test 2 muestras
Abrimos los 4 datasets de pruebas y entrenamiento
https://docs.google.com/document/d/1xV-HKNM0G5HqsYUuwWMMbefxe5-06TlboaoZ1ZouReE/edit#bookmark=id.dcefsfjx2hyp
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""
Tenemos el conjunto de entrenamiento con las métricas extremas de variables
independientes yy_train vector variable dependiente.

Abajo los conjuntos de prueba con la matriz de varaibles independientes X y el
vector variable dependiente 
Y entonces, lo que sucede es que estamos construyendo nuestro modelo 
temprano de máquina en este conjunto de tendencias estableciendo algunas 
correlaciones entre las variables independientes aquí y la variable dependiente aquí.
Y una vez que el modelo de ML entienda las correlaciones entre variables independientes
y la variable dependiente, nosotros probaremos si la maquina o el modelo puede aplicar
las correlaciones que has entendido  basándose en el conjunto de entrenamiento 
y en el conjunto de prueba
Esto significa que miraremos si podemos predecir que este registtro de 
indice cero de conjunto de pruebas X_test no va a comprar elproducto en y_test

ESto es capaz de predecirlo basado en lo que ha aprendido del conjunto de entrenamiento
Por lo tanto, cuanto mejor aprenda las correlaciones en el conjunto de entrenamiento, 
mejor será la predicción de los resultados en el conjunto de prueba.

Pero si se aprende demasiado de memoria las correlaciones de los conjuntos 
de entrenamiento, usted sabe aprendiéndolos de memoria y no entendiéndolos, 
entonces tendrá problemas para predecir lo que está sucediendo sobre el 
conjunto de pruebas, Porque como él aprende por correlaciones difíciles, 
no entendió muy bien la lógica y no podrá hacer buenas predicciones.

Esto es llamado overfitting o sobre entrenamiento lo cual se hablara en la 
seccion de regresion y aprenderemos como usar tecnicas de regularizacion para
prevenirlo

LO realmente importnae es entender que encesitamos tener dos 
diferentes datasets 
Training set con el cual el modelo de ML Aprende
Test set, sobre el cual probamos si el modelo de ML aprendió correctamente
las correlaciones

Ahora conocemos como dividir nuestro dataset dentro de un conjiunto de
entrenamiento y de pŕuebas.
ESto debe hacerse e cualquier modelo ML en donde hay que probar el rendimiento
de mi modelo con un conjunto separado de datos de pruebas
"""


# Feature Scaling
# La explicacion empieza aqui
# https://docs.google.com/document/d/1dCiLKG65mJTXyNzcBTdPzcpS9NyaL3p6nQ-snEWAY-M/edit
# MIraremos como son transformadas las variables cómo van desde
# valores grandes y muy diferentes a valores pequeños e iguales.

"""
Primero esto
https://docs.google.com/document/d/1dCiLKG65mJTXyNzcBTdPzcpS9NyaL3p6nQ-snEWAY-M/edit#bookmark=id.32u67sg0ky9r
Existen dos preguntas que podemos resolver nosotros mismos

La primera es realmente importante y podemos encontrarla mucho en 
Google. 
¿Necesitamos ajustar y transformar las variables ficticias?

Como podemos ver las variables ficticias nos permiten cambiar su formato 
y estan con valores entre 0 y 1

En Google algunos dicen que no es necesario escalar estas variables ficticias 
Otros dicen que si es necesario porque queremos precisi[on en las predicciones
En la opinion del teacher, dice que depende del contexto

Depende de que tanto deseo mantener interpretaci[on en mis modelos
Porque si escalamos esto será bueno porque todo estará en la misma 
escala, seremos felices con eso y será bueno para nuestras 
predicciones, pero quién perderá la interpretación de saber 
qué observaciones pertenecen a qué país, etc.

Así que, como quiera, no romperá su modelo si no escala las 
variables ficticias porque de hecho habrá en la misma escala 
que las escalas futuras.

Aquí las variables que tomamos van entre - 1 y 1 creo que veremos
Las variables aquí que tomamos van entre menos 1 y 1 creo que veremos, 
pero dado que este es nuestro último tutorial y no tendremos ninguna 
interpretación para hacer, escalaremos esas demasiadas variables

"""


from sklearn.preprocessing import StandardScaler

# Creamos un nuevo objeto de la clase StandardScaler()
# para escalar las variables independientes X
# Pero hasta ahora solo necesitaremos escalar las características de la matriz X
sc_X = StandardScaler()

# Y ahora, de manera muy simple, ajustaremos y transformaremos
# directamente nuestro conjunto de entrenamiento y R en un conjunto
# Vamos a transformar X_train, así que recalcularemos X_train porque
# queremos que sea de escala, y para hacerlo tomaremos nuestro objeto
# sc_X y luego llamaremos al método de fit_transform

X_train = sc_X.fit_transform(X_train)
# Es importante entender que cuando estamos aplicando nuestro
# objeto StandardScaler a nuestro conjunto de entrenamiento, es necesario
# hacer un ajuste al objeto de los conjuntos de entrenamiento y
# luego transformarlo y nos daremos cuenta que para el conjunto de pruebas
# no se realiza el mismo procedimiento porque aqui nosotros solo
# transformaremos el conjunto de pruebas
# APLICAMOS el metodo transform() y no fit_transform()
# Porque para el conjunto de entrenamiento debemos ajustarlo y luego
# transformar el conjunto de entrenamiento


# Transformamos el conjunto de pruebas, no necesita ser ajustado
# con fit_transform porque ya esta ajustado al conjunto de entrenamiento
X_test = sc_X.transform(X_test)

""" 
sigue aqui https://docs.google.com/document/d/1dCiLKG65mJTXyNzcBTdPzcpS9NyaL3p6nQ-snEWAY-M/edit#bookmark=id.uv4pocjl240d
Ahora vemos que en X_train, todas las variables pertenecen 
al mismo rango, estan en la misma escala, se puede detallar
que todas las variables estan entre -1 y 1 

ESto es perfecto, pues mejorara mucho nuestros modelos de ML

Incluso, si algunas veces los modelos de ML no son basados en 
distancias Euclidianas aún asi es muy necesario hacer
Feature scaling porque el algoritmo convergera mucho 
mas rapido- Este sera el caso para los arboles de decision, 
los cuales no son basados en Distancias Euclidianas pero
pero veremos que tendremos que hacer escalas de características porque 
si no lo hacemos se ejecutarán durante un tiempo muy largo

MIramos x_test 
sigue aqui 
https://docs.google.com/document/d/1dCiLKG65mJTXyNzcBTdPzcpS9NyaL3p6nQ-snEWAY-M/edit#bookmark=id.5ifa56o283s5


La segunda pregunta tiene que vewr con 
¿NEcesirtamos aplicar escalamiento de caracteristicas al vector de la variable
dependiente y? En este caso a y_train y a y_test?

Asi es como vemos y_train
y_train
Out[2]: array([1, 1, 1, 0, 1, 0, 0, 1])
sigue aqui https://docs.google.com/document/d/1dCiLKG65mJTXyNzcBTdPzcpS9NyaL3p6nQ-snEWAY-M/edit#bookmark=id.9fajkrondguw
"""
# Creamos un nuevo objeto de la clase StandardScaler()
# para escalar la variable vector dependiente y
"""
¿NEcesirtamos aplicar escalamiento de caracteristicas al vector de la variable
dependiente y? En este caso a y_train y a y_test
Es una variable categorica porque toma solo dos valors 
Y ahora la pregunta es ¿necesitamos aplicar funciones de escala en este?
Y la respuesta es No esta vez
No necesitamos hacerlo porque este es un problema de clasificación con una categoria llamada variable dependiente

PARA LA REGRESIÓN CUANDO la variable dependiente tomará un gran rango de valores, necesitaremos aplicar la escala de características a la variable dependiente y también
"""
# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train)"""

"""
Hemos hecho todos los pasos requeridos para el preprocesamiento de datos
los que hay que hacer para preparar cualquier dataset con el cual 
deseemos construir modelos de ML
"""