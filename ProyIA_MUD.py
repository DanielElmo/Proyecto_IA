#Proyecto Final
#Moreno Ulloa Daniel

import streamlit as st
#from PLT import image

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from apyori import apriori
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial import distance
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import model_selection
import graphviz
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import model_selection

Hipoteca = pd.read_csv('Hipoteca.csv', header=0)
Movies = pd.read_csv('Movies.csv', header=None)
StoreData = pd.read_csv('store_data.csv', header=0)
WDBCOriginal = pd.read_csv('WDBCOriginal.csv', header=0)

def RAsociacion():
    #Archivos soportados: Movies y Store_Data
    Data=Movies
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
    ListaD = Data.stack().groupby(level=0).apply(list).tolist()
    ReglasC = apriori(ListaD,min_support=0.01,min_confidence=0.3,min_lift=2) #Datos del usuario
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
    
def MDEuclidiana():
    #Archivos soportados: Hipoteca y WDBCOriginal
    Data=Hipoteca
    Objeto1 = Data.iloc[100] #Objetos elegidos por el usuario.
    Objeto2 = Data.iloc[200]
    #Distancia Euclideana
    DstEuclidiana = cdist(Data, Data, metric='euclidean')
    MEuclidiana = pd.DataFrame(DstEuclidiana)
    print(MEuclidiana)
    dstEuclidiana = distance.euclidean(Objeto1,Objeto2)
    print(dstEuclidiana)
    
def MDChebyshev():
    Data=Hipoteca
    Objeto1 = Data.iloc[100] #Objetos elegidos por el usuario.
    Objeto2 = Data.iloc[200]
    #Distancia Chebyshev
    DstCheb = cdist(Hipoteca, Hipoteca, metric='chebyshev')
    MCheb = pd.DataFrame(DstCheb)
    print(MCheb)
    dstCheb = distance.chebyshev(Objeto1,Objeto2)
    print(dstCheb)
    
def MDManhattan():
    Data=Hipoteca
    Objeto1 = Data.iloc[100] #Objetos elegidos por el usuario.
    Objeto2 = Data.iloc[200]
    #Distancia Manhattan
    DstCity = cdist(Hipoteca, Hipoteca, metric='cityblock')
    MCity = pd.DataFrame(DstCity)
    print(MCity)
    dstCity = distance.cityblock(Objeto1,Objeto2)
    print(dstCity)
    
def MDMinkowsky():
    Data=Hipoteca
    Objeto1 = Data.iloc[100] #Objetos elegidos por el usuario.
    Objeto2 = Data.iloc[200]
    #Distancia Minkowsky
    DstMin = cdist(Hipoteca, Hipoteca, metric='minkowski',p=1.5)
    MMin = pd.DataFrame(DstMin)
    print(MMin)
    dstMin = distance.minkowski(Objeto1,Objeto2, p=1.5)
    #Comparación de Distancias
    print(dstMin)
    
def MDistancia():
    met = st.selectbox("Escoja la metrica de distancia",
                       ['Euclidiana','Chebyshev','Manhattan','Minkowsky'])
    if(met=='Euclidiana'):
        MDEuclidiana()
    elif(met=='Chebyshev'):
        MDChebyshev()
    elif(met=='Manhattan'):
        MDManhattan()
    elif(met=='Minkowsky'):
        MDMinkowsky()
    else:
        st.error('Error')
        
    
def CJerarquico():
    #Archivos soportados: Hipoteca y WDBCOriginal
    Data=Hipoteca
    sns.scatterplot(x='ahorros', y ='ingresos', data=Data)
    plt.title('Gráfico de dispersión')
    plt.xlabel('Ahorros')
    plt.ylabel('Ingresos')
    plt.show()
    MatrizHipoteca = np.array(Data[['ingresos', 'gastos_comunes', 'pago_coche', 'gastos_otros', 'ahorros', 'vivienda', 'estado_civil', 'hijos', 'trabajo']])
    estandarizar = StandardScaler()
    MEstandarizada = estandarizar.fit_transform(MatrizHipoteca)
    pd.DataFrame(MEstandarizada)
    plt.figure(figsize=(10, 7))
    plt.title("Casos de hipoteca")
    plt.xlabel('Hipoteca')
    plt.ylabel('Distancia')
    Arbol = shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric='euclidean'))
    MJerarquico = AgglomerativeClustering(n_clusters=7, linkage='complete', affinity='euclidean')
    MJerarquico.fit_predict(MEstandarizada)
    Data = Data.drop(columns=['comprar'])
    Data['clusterH'] = MJerarquico.labels_
    CentroidesH = Data.groupby('clusterH').mean()
    CentroidesH
    
def CParticional():
    #Archivos soportados: Hipoteca y WDBCOriginal
    Data=Hipoteca
    sns.scatterplot(x='ahorros', y ='ingresos', data=Hipoteca, hue='comprar')
    plt.title('Gráfico de dispersión')
    plt.xlabel('Ahorros')
    plt.ylabel('Ingresos')
    plt.show()
    MatrizHipoteca = np.array(Data[['ingresos', 'gastos_comunes', 'pago_coche', 'gastos_otros', 'ahorros', 'vivienda', 'estado_civil', 'hijos', 'trabajo']])
    estandarizar = StandardScaler()
    MEstandarizada = estandarizar.fit_transform(MatrizHipoteca)
    pd.DataFrame(MEstandarizada)
    SSE = []
    for i in range(2, 12):
        km = KMeans(n_clusters=i, random_state=0)
        km.fit(MEstandarizada)
        SSE.append(km.inertia_)
    plt.figure(figsize=(10, 7))
    plt.plot(range(2, 12), SSE, marker='o')
    plt.xlabel('Cantidad de clusters *k*')
    plt.ylabel('SSE')
    plt.title('Elbow Method')
    plt.show()
    kl = KneeLocator(range(2, 12), SSE, curve="convex", direction="decreasing")
    plt.style.use('ggplot')
    kl.plot_knee()
    MParticional = KMeans(n_clusters=4, random_state=0).fit(MEstandarizada)
    MParticional.predict(MEstandarizada)
    Data = Data.drop(columns=['comprar'])
    Data['clusterP'] = MParticional.labels_
    CentroidesP = Data.groupby('clusterP').mean()
    CentroidesP
    
def Clustering():
    met = st.selectbox("Escoja el metodo de clustering",
                       ['Jerarquico','Particional'])
    if(met=='Jerarquico'):
        CJerarquico()
    elif(met=='Particional'):
        CParticional()
    else:
        st.error('Error')
    
def RLogistica():
    #Archivos soportados: Hipoteca y WDBCOriginal
    Data=WDBCOriginal
    Data = Data.replace({'M': 0, 'B': 1})
    X = np.array(Data[['Texture', 'Area', 'Smoothness', 'Compactness', 'Symmetry', 'FractalDimension']])
    Y = np.array(Data[['Diagnosis']])
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 0, shuffle = True)
    Clasificacion = linear_model.LogisticRegression()
    Clasificacion.fit(X_train, Y_train)
    Y_Clasificacion = Clasificacion.predict(X_validation)
    Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), Y_Clasificacion, rownames=['Real'], colnames=['Clasificación']) 
    Matriz_Clasificacion
    print("Exactitud", Clasificacion.score(X_validation, Y_validation))
    print(classification_report(Y_validation, Y_Clasificacion))
    print("Intercept:", Clasificacion.intercept_)
    print('Coeficientes: \n', Clasificacion.coef_) 
    PacienteID = pd.DataFrame({'Texture': [10.38], 'Area': [1001.0], 'Smoothness': [0.11840], 'Compactness': [0.27760], 'Symmetry': [0.2419], 'FractalDimension': [0.07871]})
    Clasificacion.predict(PacienteID)
    
def ADPronostico():
    #Archivos soportados: WDBCOriginal
    Data=WDBCOriginal
    X = np.array(Data[['Texture', 'Perimeter', 'Smoothness', 'Compactness', 'Symmetry', 'FractalDimension']])
    Y = np.array(Data[['Area']])
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 0, shuffle = True)
    PronosticoAD = DecisionTreeRegressor()
    PronosticoAD.fit(X_train, Y_train)
    Y_Pronostico = PronosticoAD.predict(X_test)
    print('Criterio: \n', PronosticoAD.criterion)
    print('Importancia variables: \n', PronosticoAD.feature_importances_)
    print("MAE: %.4f" % mean_absolute_error(Y_test, Y_Pronostico))
    print("MSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico))
    print("RMSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico, squared=False))
    print('Score (Bondad de ajuste): %.4f' % r2_score(Y_test, Y_Pronostico))
    Importancia = pd.DataFrame({'Variable': list(Data[['Texture', 'Perimeter', 'Smoothness',	'Compactness', 'Symmetry', 'FractalDimension']]), 'Importancia': PronosticoAD.feature_importances_}).sort_values('Importancia', ascending=False)
    print(Importancia)
    Elementos = export_graphviz(PronosticoAD, feature_names = ['Texture', 'Perimeter', 'Smoothness', 'Compactness', 'Symmetry', 'FractalDimension'])
    Arbol = graphviz.Source(Elementos)
    Arbol
    AreaTumorID = pd.DataFrame({'Texture': [10.38], 'Perimeter': [122.8], 'Smoothness': [0.11840], 'Compactness': [0.27760], 'Symmetry': [0.2419], 'FractalDimension': [0.07871]})
    PronosticoAD.predict(AreaTumorID)
    
def ADClasificacion():
    #Archivos soportados: WDBCOriginal
    Data=WDBCOriginal
    Data = Data.replace({'M': 'Malignant', 'B': 'Benign'})
    X = np.array(Data[['Texture', 'Area', 'Smoothness', 'Compactness', 'Symmetry', 'FractalDimension']])
    Y = np.array(Data[['Diagnosis']])
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 0, shuffle = True)
    ClasificacionAD = DecisionTreeClassifier()
    ClasificacionAD.fit(X_train, Y_train)
    Y_Clasificacion = ClasificacionAD.predict(X_validation)
    print('Criterio: \n', ClasificacionAD.criterion)
    print('Importancia variables: \n', ClasificacionAD.feature_importances_)
    print("Exactitud", ClasificacionAD.score(X_validation, Y_validation))
    print(classification_report(Y_validation, Y_Clasificacion))
    Importancia = pd.DataFrame({'Variable': list(Data[['Texture', 'Area', 'Smoothness', 'Compactness', 'Symmetry', 'FractalDimension']]), 'Importancia': ClasificacionAD.feature_importances_}).sort_values('Importancia', ascending=False)
    print(Importancia)
    Elementos = export_graphviz(ClasificacionAD, feature_names = ['Texture', 'Area', 'Smoothness', 'Compactness', 'Symmetry', 'FractalDimension'], class_names = Y_Clasificacion)  
    Arbol = graphviz.Source(Elementos)
    Arbol
    PacienteID = pd.DataFrame({'Texture': [10.38], 'Area': [1001.0], 'Smoothness': [0.11840], 'Compactness': [0.27760], 'Symmetry': [0.2419], 'FractalDimension': [0.07871]})
    ClasificacionAD.predict(PacienteID)
    
def ADecision():
    met = st.selectbox("Escoja el modelo del arbol",
                       ['Pronostico','Clasificacion'])
    if(met=='Pronostico'):
        ADPronostico()
    elif(met=='Clasificacion'):
        ADClasificacion()
    else:
        st.error('Error')

def main():
    st.title("Inteligencia Artificial")
    if(st.button("Reglas de Asociasion")):
        RAsociacion()
    elif(st.button("Metricas de Distancia")):
        MDistancia()
    elif(st.button("Clustering")):
        Clustering()
    elif(st.button("Regresion Logistica")):
        RLogistica()
    elif(st.button("Arboles de Decision")):
        ADecision()        

    
main();
    