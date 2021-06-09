import matplotlib.pyplot as plt
from random import uniform, random
import random
from cv2 import cv2
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations
from math import sqrt, fabs
import borrar
from time import time
from multiprocessing import Pool
######################## funciones ##############################
##################### particulas ##########################
n = 150
x = np.random.uniform(20, 80, n)
y = np.random.uniform(20, 80, n)
z = np.random.uniform(20, 80, n)
ct = np.random.uniform(0, 10, n)
p = pd.DataFrame({'x': x, 'y': y, 'z': z, 'ct':ct})
################### particulas dopaje#######################
porc = 40 
m = round((porc*n)/100)
xcer= np.random.uniform(20, 80, m)
ycer= np.random.uniform(20, 80, m)
xpol= np.random.uniform(20, 80, m)
ypol= np.random.uniform(20, 80, m)
xplata= np.random.uniform(20, 80, m)
yplata= np.random.uniform(20, 80, m)
p2= pd.DataFrame({'xcer': xcer, 'ycer': ycer, 'xpol': xpol,
                  'ypol': ypol, 'xplata': xplata, 'yplata': yplata})
######################### VARIABLES ########################
fusion=90
afecta=8  #distancia a la que afecta la particula
afecta2=10 #distancia a la que afecta el dopaje
masa=0.05
mejores=[]
#######################  CICLOS VARIA TEMPERATURA Y MUESTRAS ################
temperatura=[]
MB, CR, PO, PL=[], [], [], []
for temp in range(45, 250,5):
    dist=[]
    for mezcla in ([0,0,0],[p2.xcer,p2.ycer,.1],[p2.xpol,p2.ypol,.06],
                   [p2.xplata,p2.yplata,.03]):
        RESUL= funciones.calentamiento(mezcla,p,p2,afecta,afecta2,temp,m,n,masa)    
        dist.append(RESUL)                 
    temperatura.append(temp)
    mejores.append(dist)

    MB.append(min(dist[0]))
    CR.append(min(dist[1]))
    PO.append(min(dist[2]))
    PL.append(min(dist[3]))
##### Gráficas
fig = plt.figure()
plt.xlabel('Temperatura')
plt.ylabel('Cercania a punto de fusión')
plt.scatter(temperatura, MB, color='blue', s=10)
plt.plot(temperatura, MB, color='blue', label='Material puro')
plt.scatter(temperatura, CR, color='red', s=10)
plt.plot(temperatura, CR, color='red',label='Aleacion cerámica')
plt.scatter(temperatura, PO, color='g', s=10)
plt.plot(temperatura, PO, color='g',label='Aleacion polimérica')
plt.scatter(temperatura, PL, color='purple', s=10)
plt.plot(temperatura, PL, color='purple',label='Aleacion plata')
plt.legend()
plt.legend(loc='upper right', title='Aleaciones')
plt.savefig("temp_vari.png")
plt.show()
plt.close()

fig = plt.figure(figsize=(7,8))
box=plt.boxplot([MB, CR, PO, PL], patch_artist=True)
colors = ['blue', 'red', 'green', 'purple']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
plt.xlabel('Aleaciones')
plt.xticks([1,2,3,4],['Puro','Cerámico','Polímero','Plata'])
plt.xticks(rotation=45)
plt.ylabel('Distancias cerca a punto de fusión')
plt.savefig("CB_tmp_vari.png")
plt.show()
plt.close()

######################### CICLO VARÍA MASA #################################
temp=45
masa_inc=[]
antes = time()    
MB2, CR2, PO2, PL2=[], [], [], []
for masa in range(5, 31):
    masa = masa/100
    dist=[]
    for mezcla in ([0,0,0],[p2.xcer,p2.ycer,.1],[p2.xpol,p2.ypol,.06],
                   [p2.xplata,p2.yplata,.03]):
        result=funciones.calentamiento(mezcla,p,p2,afecta,afecta2,temp,m,n,masa)    
        dist.append(result)
    masa_inc.append(masa)
    MB2.append(min(dist[0]))
    CR2.append(min(dist[1]))
    PO2.append(min(dist[2]))
    PL2.append(min(dist[3]))
print('puro T/M',MB2)
print('cer T/M',CR2)
print('pol T/M',PO2)
print('plt T/M',PL2)
fig = plt.figure()
plt.xlabel('Variación de masa')
plt.ylabel('Cercanía a punto de fusión')
plt.scatter(masa_inc, MB2, color='blue', s=10)
plt.plot(masa_inc, MB2, color='blue', label='Material puro')
plt.scatter(masa_inc, CR2, color='red', s=10)
plt.plot(masa_inc, CR2, color='red',label='Aleación cerámica')
plt.scatter(masa_inc, PO2, color='g', s=10)
plt.plot(masa_inc, PO2, color='g',label='Aleación polimérica')
plt.scatter(masa_inc, PL2, color='purple', s=10)
plt.plot(masa_inc, PL2, color='purple',label='Aleación plata')
plt.legend()
plt.legend(loc='lower right', title='Aleaciones')
plt.savefig("Tmp_masa_vari.png")
plt.show()
plt.close()


fig = plt.figure(figsize=(7,8))
box2=plt.boxplot([MB2, CR2, PO2, PL2], patch_artist=True)
colors = ['blue', 'red', 'green', 'purple']
for patch, color in zip(box2['boxes'], colors):
    patch.set_facecolor(color)
plt.xlabel('Aleación')
plt.xticks([1,2,3,4],['Puro','Cerámico','Polímero','Plata'])
plt.xticks(rotation=45)
plt.ylabel('Distancias cerca a punto de fusión')
plt.savefig("CB_tmp.png")
plt.show()
plt.close()

###################### Ciclo corriente respecto a masa##########

temp=100
masa_inc=[]
MB3, CR3, PO3, PL3=[], [], [], []
for masa in range(5, 31):
    masa = masa/100
    dist=[]
    for mezcla in ([0,0,0],[p2.xcer,p2.ycer,.01],[p2.xpol,p2.ypol,.04],
                   [p2.xplata,p2.yplata,.1]):
        RESUL= funciones.flujo_elec(mezcla,p,p2,afecta,afecta2,temp,m,n,masa)    
        dist.append(RESUL)
    masa_inc.append(masa)
    MB3.append(min(dist[0]))
    CR3.append(min(dist[1]))
    PO3.append(min(dist[2]))
    PL3.append(min(dist[3]))
print('puro E/M',MB3)
print('cer E/M',CR3)
print('pol E/M',PO3)
print('plt E/M',PL3)
fig = plt.figure()
plt.xlabel('Variación de masa')
plt.ylabel('Flujo electrico total')    
plt.scatter(masa_inc, MB3, color='blue', s=10)
plt.plot(masa_inc, MB3, color='blue', label='Material puro')
plt.scatter(masa_inc, CR3, color='red', s=10)
plt.plot(masa_inc, CR3, color='red',label='Aleación cerámica')
plt.scatter(masa_inc, PO3, color='g', s=10)
plt.plot(masa_inc, PO3, color='g',label='Aleación polimérica')
plt.scatter(masa_inc, PL3, color='purple', s=10)
plt.plot(masa_inc, PL3, color='purple',label='Aleación plata')
plt.legend()
plt.legend(loc='lower right', title='Aleaciones')
plt.savefig("elc_masa_vari.png")
plt.show()
plt.close()


fig = plt.figure(figsize=(7,8))
box3=plt.boxplot([MB3, CR3, PO3, PL3], patch_artist=True)
colors = ['blue', 'red', 'green', 'purple']
for patch, color in zip(box3['boxes'], colors):
    patch.set_facecolor(color)
plt.xlabel('Aleación')
plt.xticks([1,2,3,4],['Material puro','Cerámico','Polímero','Plata'])
plt.xticks(rotation=45)
plt.ylabel('Flujo eléctrico total')
plt.savefig("CB_elc.png")
plt.show()
plt.close()



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(MB2, MB3, masa_inc, s=8)
ax.plot3D(MB2, MB3, masa_inc, 'blue')
ax.scatter(CR2, CR3, masa_inc, s=8)
ax.plot3D(CR2, CR3, masa_inc, 'red')
ax.scatter(PO2, PO3, masa_inc, s=8)
ax.plot3D(PO2, PO3, masa_inc, 'green')
ax.scatter(PL2, PL3, masa_inc, s=8)
ax.plot3D(PL2, PL3, masa_inc, 'purple')
#ax.set_zticks([0.1, 0.3, 0.5],['5','50','100'])
ax.set_xlabel("Punto de fusión en temperatura")
ax.set_ylabel("Flujo de corriente")
ax.set_zlabel("Variación de masa")
plt.show()




