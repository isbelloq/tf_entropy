#Carga de librerias
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#Configuracion para uso de LaTeX en matplotlib 
plt.rcParams['text.usetex'] = True

#Path de almacenamiento de figuras
path_fig = os.environ["FIGURE_PATH"]

#Creacion de puntos
X = np.linspace(0,1,100)
Y = -((X*np.log2(X)) + ((1-X)*np.log2(1-X)))

# +
#Grafica de entropia en funcion de la probabilidad
fig, ax = plt.subplots()
ax.plot(X,Y, color='black')
ax.set_xlabel(r"$p$")
ax.set_ylabel(r"$H\left(p\right)$")

#Almacenamiento
fig.savefig(f'{path_fig}/entropia/Hvsp.pdf')
