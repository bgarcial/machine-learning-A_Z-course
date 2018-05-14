# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Mall dataset with pandas
'''
Existe un mall grande en una ciudad que contiene
información de sus clientes, los que estan suscritos
a su tarjeta de membresía. 

Cuando un cliente se suscribe proporciona 
'''
dataset = pd.read_csv('../../data/Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
# y = dataset.iloc[:, 3].values