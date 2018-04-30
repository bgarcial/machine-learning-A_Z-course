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