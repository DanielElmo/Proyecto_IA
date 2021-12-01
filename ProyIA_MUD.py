#Proyecto Final

pip install apyori

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from apyori import apriori

Hipoteca = pd.read_csv('Hipoteca.csv', header=0)
Movies = pd.read_csv('Movies.csv', header=0)
RGeofisicos = pd.read_csv('RGeofisicos.csv', header=0)
StoreData = pd.read_csv('store_data.csv', header=0)
WDBCOriginal = pd.read_csv('WDBCOriginal.csv', header=0)
