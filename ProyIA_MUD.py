# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 22:22:32 2021

@author: 51716
"""

#Proyecto Final
#Moreno Ulloa Daniel

#pip install apyori
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from apyori import apriori

Hipoteca = pd.read_csv('Hipoteca.csv', header=0)
Movies = pd.read_csv('Movies.csv', header=None)
RGeofisicos = pd.read_csv('RGeofisicos.csv', header=0)
StoreData = pd.read_csv('store_data.csv', header=0)
WDBCOriginal = pd.read_csv('WDBCOriginal.csv', header=0)

def RAsociacion():
  Data=Movies #o Data=StoreData
  Transacciones = Data.values.reshape(149200).tolist()
  ListaM = pd.DataFrame(Transacciones)
  ListaM['Frecuencia'] = 0
  ListaM = ListaM.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True)
  ListaM['Porcentaje'] = (ListaM['Frecuencia'] / ListaM['Frecuencia'].sum())
  ListaM = ListaM.rename(columns={0 : 'Item'})
  plt.figure(figsize=(16,20), dpi=300)
  plt.ylabel('Item')
  plt.xlabel('Frecuencia')
  plt.barh(ListaM['Item'], width=ListaM['Frecuencia'], color='blue')
  plt.show()
  MoviesLista = Data.stack().groupby(level=0).apply(list).tolist()
  ReglasC = apriori(MoviesLista,min_support=0.01,min_confidence=0.3,min_lift=2) #Datos del usuario
  ResultadosC = list(ReglasC)
  for item in ResultadosC:
    #El primer índice de la lista
    #Emparejar = item[0]
    #items = [x for x in Emparejar]
    print("Regla: " + str(item[0]))
    print("Soporte: " + str(item[1]))
    print("Confianza: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3])) 
    print("-------------------------------------")
    
#def MDistancia():
  
