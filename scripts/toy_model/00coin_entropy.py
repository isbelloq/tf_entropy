# -*- coding: utf-8 -*-
# +
#Carga de librerias
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#Configuracion para uso de LaTeX en matplotlib 
plt.rcParams['text.usetex'] = True
# -

#Path de almacenamiento de figuras
path_fig = os.environ["FIGURE_PATH"]

# +
#Creacion de puntos
X = np.linspace(0,1,100)

#Calculo de la informacion
I_p = -np.log2(X)
I_q = -np.log2(1-X)

#Calculo de la entropia
Y = X*I_p + (1-X)*I_q

# +
#Grafica de entropia en funcion de la probabilidad
fig, ax = plt.subplots()
ax.plot(X,Y, color='black')
ax.set_xlabel(r"$p$")
ax.set_ylabel(r"$H\left(p\right)$")

#Almacenamiento
fig.savefig(f'{path_fig}/entropia/Hvsp.pdf')

# +
fig, ax = plt.subplots()

#Grafica de la infromación I_p en funcion de la probabilidad
ax.plot(X,I_p, color='black', label = r"Información del lanzamiento de cara $I_p$")
#Grafica de la infromación I_q en funcion de la probabilidad
ax.plot(X,I_q, color='red', label = r"Información del lanzamiento de sello $I_q$")

ax.set_xlabel(r"$p$")
ax.set_ylabel(r"$I$")
ax.legend()

#Almacenamiento
fig.savefig(f'{path_fig}/entropia/Ivsp.pdf')
