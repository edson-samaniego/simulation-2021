import matplotlib.pyplot as plt
from random import uniform, random
import random
from cv2 import cv2
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations
from math import sqrt, fabs

#################### particulas de material base
n = 80
x = np.random.uniform(20, 80, n)
y = np.random.uniform(20, 80, n)
z = np.random.uniform(20, 80, n)
ct = np.random.uniform(0, 10, n)
p = pd.DataFrame({'x': x, 'y': y, 'z': z, 'ct':ct})

#################### particula de calor distancia al centro 
fusion=90
afecta=5  #distancia eucl a la que afecta la particula

temp=45 
calorx=np.random.uniform(0, 20, 1)
calory=np.random.uniform(0, 100, 1)

dis_cen= sqrt((50-calorx)**2+(50-calory)**2) # distancia al centro 
print(dis_cen)
################# calor distancias con todas las particulas
DE= []
for i in range(0,n):
    fila= p.iloc[i] #cambio con for la lista por fila 
    xp=fila.x
    yp=fila.y
    ctp=fila.ct
    eucl=sqrt((xp-calorx)**2+(yp-calory)**2)# distancias de cada particula
    DE.append(eucl)
    
if (any([d<afecta for d in DE]))== True:
    cercanos=sum([d<afecta for d in DE])
    print(cercanos,'cercanos a la particula para afectar propagacion')

##### modificando
calorx_2=(50-calorx)
calory_2=(50-calory)
print(calorx, calory)
print(calorx_2, calory_2)
######################## gráfico 2d
r = [0, 100]
fig = plt.figure()
LX=[20, 20, 80, 80]
LY=[20, 80, 80, 20]

plt.xlim([0, 100])
plt.ylim([0, 100])
plt.title('posicion inicial')
plt.plot(LX, LY, color='blue')
plt.plot((LX[0],LX[3]), (LY[0],LY[3]), color='blue')
plt.scatter(p.x, p.y, color="g", s=20)
plt.scatter((r[1])/2, (r[1])/2, color="r", s=5)
plt.scatter(calorx, calory, color="r", s=25)
plt.show()















