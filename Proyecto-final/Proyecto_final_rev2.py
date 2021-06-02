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
n = 150
x = np.random.uniform(20, 80, n)
y = np.random.uniform(20, 80, n)
z = np.random.uniform(20, 80, n)
ct = np.random.uniform(0, 10, n)
p = pd.DataFrame({'x': x, 'y': y, 'z': z, 'ct':ct})
fusion=90
afecta=7  #distancia a la que afecta la particula
afecta2=6 #distancia a la que afecta el dopaje
temp=40
######################### DATAFRAME DE PARTICULAS DE DOPAJE
porc = 20 #porcentaje que se aplicara de dopaje o particulas 
m = round((porc*n)/100)
xcer= np.random.uniform(20, 80, m)
ycer= np.random.uniform(20, 80, m)
xpol= np.random.uniform(20, 80, m)
ypol= np.random.uniform(20, 80, m)
xplata= np.random.uniform(20, 80, m)
yplata= np.random.uniform(20, 80, m)
p2= pd.DataFrame({'xcer': xcer, 'ycer': ycer, 'xpol': xpol,
                  'ypol': ypol, 'xplata': xplata, 'yplata': yplata})

#################### posicion inicial ############################
for mezcla in ([0,0,0],[p2.xcer,p2.ycer,.1],[p2.xpol,p2.ypol,.06],
               [p2.xplata,p2.yplata,.03]):
    calorx=0
    calory=0
    trans_calor= mezcla[2]
    r = [0, 100]
    fig = plt.figure()
    LX=[20, 20, 80, 80]
    LY=[20, 80, 80, 20]
    plt.scatter(mezcla[0], mezcla[1], marker="^", color="orange", s=10)
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.title('posicion inicial')
    plt.plot(LX, LY, color='y')
    plt.plot((LX[0],LX[3]), (LY[0],LY[3]), color='y')
    plt.scatter(p.x, p.y, color="g", s=20)
    plt.scatter((r[1])/2, (r[1])/2, marker="+", color="r", s=5)
    plt.scatter(calorx, calory, color="r", s=25)
    plt.savefig('p_inicial.png')

################# calor distancias con todas las particulas############################
    dist_centro=[]
    for img in range(0,50):
        DE= []
        DE2= []
        cercanos=0
        dopaje=0
        for i in range(0,n):
            fila= p.iloc[i] #cambio con for la lista por fila 
            xp=fila.x
            yp=fila.y
            eucl=sqrt((xp-calorx)**2+(yp-calory)**2)# distancias de cada particula
            DE.append(eucl)
        if (any([d<afecta for d in DE]))== True:
            cercanos=sum([d<afecta for d in DE])

            
        if mezcla[2] == .1:    
            for j in range(0,m):
                fila2= p2.iloc[j] #cambio con for la lista por fila
                xp2=fila2.xcer
                yp2=fila2.ycer
                eucl2=sqrt((xp2-calorx)**2+(yp2-calory)**2)# distancias de cada particula
                DE2.append(eucl2)
            if (any([e<afecta2 for e in DE2]))== True:
                dopaje=sum([e<afecta2 for e in DE2])
        if mezcla[2] ==.06:    
            for j in range(0,m):
                fila2= p2.iloc[j] #cambio con for la lista por fila
                xp2=fila2.xpol
                yp2=fila2.ypol
                eucl2=sqrt((xp2-calorx)**2+(yp2-calory)**2)# distancias de cada particula
                DE2.append(eucl2)
            if (any([e<afecta2 for e in DE2]))== True:
                dopaje=sum([e<afecta2 for e in DE2])
        if mezcla[2] == .03:    
            for j in range(0,m):
                fila2= p2.iloc[j] #cambio con for la lista por fila
                xp2=fila2.xplata
                yp2=fila2.yplata
                eucl2=sqrt((xp2-calorx)**2+(yp2-calory)**2)# distancias de cada particula
                DE2.append(eucl2)
            if (any([e<afecta2 for e in DE2]))== True:
                dopaje=sum([e<afecta2 for e in DE2])
            
 
##################################### modificando#################################
        print(cercanos,'particulas de base')
        print(dopaje,'particulas agregadas dopaje')    
        if cercanos==0:
            calorx=(calorx+1+(temp*.003))if calorx< 50 else(calorx-1-(temp*.003))
            calory=(calory+1+(temp*.003))if calory< 50 else(calory-1-(temp*.003))        
###########################################################
        if cercanos >0:
            factor= cercanos * .05
            calorx=((calorx+1+(temp*.003))-factor)if calorx< 50 else((calorx-1-(temp*.003))+factor)
            calory=((calory+1+(temp*.003))-factor)if calory< 50 else((calory-1-(temp*.003))+factor)
            if dopaje > 0:
                factor2= dopaje * trans_calor
                calorx=(calorx-factor2) if calorx< 50 else (calorx+factor2)
                calory=(calory-factor2) if calorx< 50 else (calory+factor2)
            
        mejor=[(sqrt((50-calorx)**2+(50-calory)**2)),img]
        dist_centro.append(mejor)
############################################################
        fig = plt.figure()
        plt.xlim([0, 100])
        plt.ylim([0, 100])
        plt.plot(LX, LY, color='y')
        plt.plot((LX[0],LX[3]), (LY[0],LY[3]), color='y')
        plt.scatter(p.x, p.y, color="g", s=20)
        plt.scatter((r[1])/2, (r[1])/2, marker="+", color="r", s=5)
        plt.scatter(calorx, calory,marker="+", color="r", s=25)
        plt.scatter(mezcla[0], mezcla[1], marker="^", color="orange", s=10)
        fig.suptitle('Paso {:d}'.format(img + 1))
        plt.savefig('p2_t{:d}_p.png'.format(img + 1))
    
    print(dist_centro)
