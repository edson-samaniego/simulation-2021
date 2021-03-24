from cv2 import cv2
import pandas as pd 
import matplotlib.pyplot as plt
from math import floor, log, sqrt
from random import random, uniform
import numpy as np
 
l = 1.5
n = 50
pi = 0.05
pr = 0.02 # prob. de recuperar
v = l / 30
r = 0.1
tmax = 100
digitos = floor(log(tmax, 10)) + 1
d0,d1,d2,d3,d4,d5,d6,d7,d8,d9 = [],[],[],[],[],[],[],[],[],[]
n0,n1,n2,n3,n4,n5,n6,n7,n8,n9 = [],[],[],[],[],[],[],[],[],[]
p0,p1,p2,p3,p4,p5,p6,p7,p8,p9 = [],[],[],[],[],[],[],[],[],[]

c = {'I': 'r', 'S': 'g', 'R': 'orange', 'V':'blue'}
m = {'I': 'o', 'S': 's', 'R': '2', 'V':'P'}
replicas = 25
for pv in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    for rep in range(replicas):
        agentes =  pd.DataFrame()
        agentes['x'] = [uniform(0, l) for i in range(n)]
        agentes['y'] = [uniform(0, l) for i in range(n)]
        agentes['dx'] = [uniform(-v, v) for i in range(n)]
        agentes['dy'] = [uniform(-v, v) for i in range(n)]        
        agentes['estado'] = ['V' if random() < pv else 'S' if random() > pi else 'I' for i in range(n)]
        epidemia = []
####################### iteraciones o imagenes que son 50
        for tiempo in range(tmax):
            conteos = agentes.estado.value_counts()
            infectados = conteos.get('I', 0)
            epidemia.append(infectados)
            if infectados == 0:
                break
            contagios = [False for i in range(n)]      
#############################  ciclo donde ve los infectados    
            for i in range(n): # contagios
                a1 = agentes.iloc[i]# revisa agente por agente segun el ciclo
                if a1.estado == 'I':  # en la revision si encuentra un infectado
                    for j in range(n): #entra a un ciclo que revisa de nuevo los agentes
                        a2 = agentes.iloc[j]
                        if a2.estado == 'S': #si el agente es susceptible 
                            d = sqrt((a1.x - a2.x)**2 + (a1.y - a2.y)**2) # obtiene la euclidiana de el infectado a el susceptible
                            if d < r:   # si la distancia es menor al rango .1 
                                if random() < (r - d) / r: # condicion para ver si infecta
                                    contagios[j] = True # el mas cercano cambia a true

            for i in range(n): # ciclo de movimientos
                a = agentes.iloc[i] # revisa de nuevo los agentes
                if contagios[i]:  # si esta en los contagios 
                    agentes.at[i, 'estado'] = 'I' # actualiza el estado de S a I
                elif a.estado == 'I': #si ya estab infectado desde antes
                    if random() < pr: # entonces hace un random de si es menor a pr
                        agentes.at[i, 'estado'] = 'R' # el agente cambiara a R
        ### actualiza nuevos cambios        
                x = a.x + a.dx
                y = a.y + a.dy
                x = x if x < l else x - l
                y = y if y < l else y - l
                x = x if x > 0 else x + l
                y = y if y > 0 else y + l
                agentes.at[i, 'x'] = x
                agentes.at[i, 'y'] = y
            n0.append([epidemia[tiempo], (tiempo)])

            
############## Para obtener gráficos #################            
     
        if pv == 0:
            d0.append(epidemia)
            if epidemia != [0]:
                if max(n0):
                    a=max(n0)
                    p0.append(a[1])
                
        if pv == 0.1:
            d1.append(epidemia)
            if epidemia != [0]:
                if max(n0):
                    b=max(n0)
                    p1.append(b[1])
                
        if pv == 0.2:
            d2.append(epidemia)
            if epidemia != [0]:
                if max(n0):
                    c=max(n0)
                    p2.append(c[1])
                
        if pv == 0.3:
            d3.append(epidemia)
            if epidemia != [0]:
                if max(n0):
                    d=max(n0)
                    p3.append(d[1])
                
        if pv == 0.4:
            d4.append(epidemia)
            if epidemia != [0]:
                if max(n0):
                    e=max(n0)
                    p4.append(e[1])
                
        if pv == 0.5:
            d5.append(epidemia)
            if epidemia != [0]:
                if max(n0):
                    f=max(n0)
                    p5.append(f[1])
                
        if pv == 0.6:
            d6.append(epidemia)
            if epidemia != [0]:
                if max(n0):
                    g=max(n0)
                    p6.append(g[1])
                
        if pv == 0.7:
            d7.append(epidemia)
            if epidemia != [0]:
                if max(n0):
                    h=max(n0)
                    p7.append(h[1])
                
        if pv == 0.8:
            d8.append(epidemia)
            if epidemia != [0]:
                if max(n0):
                    i=max(n0)
                    p8.append(i[1])
                
        if pv == 0.9:
            d9.append(epidemia)
            if epidemia != [0]:
                if max(n0):
                    j=max(n0)
                    p9.append(j[1])                
        n0.clear()
         
D0=(d0[0]+d0[1]+d0[2]+d0[3]+d0[4]+d0[5]+d0[6]+d0[7]+d0[8]+d0[9])
D1=(d1[0]+d1[1]+d1[2]+d1[3]+d1[4]+d1[5]+d1[6]+d1[7]+d1[8]+d1[9])
D2=(d2[0]+d2[1]+d2[2]+d2[3]+d2[4]+d2[5]+d2[6]+d2[7]+d2[8]+d2[9])
D3=(d3[0]+d3[1]+d3[2]+d3[3]+d3[4]+d3[5]+d3[6]+d3[7]+d3[8]+d3[9])
D4=(d4[0]+d4[1]+d4[2]+d4[3]+d4[4]+d4[5]+d4[6]+d4[7]+d4[8]+d4[9])
D5=(d5[0]+d5[1]+d5[2]+d5[3]+d5[4]+d5[5]+d5[6]+d5[7]+d5[8]+d5[9])
D6=(d6[0]+d6[1]+d6[2]+d6[3]+d6[4]+d6[5]+d6[6]+d6[7]+d6[8]+d6[9])
D7=(d7[0]+d7[1]+d7[2]+d7[3]+d7[4]+d7[5]+d7[6]+d7[7]+d7[8]+d7[9])
D8=(d8[0]+d8[1]+d8[2]+d8[3]+d8[4]+d8[5]+d8[6]+d8[7]+d8[8]+d8[9])
D9=(d9[0]+d9[1]+d9[2]+d9[3]+d9[4]+d9[5]+d9[6]+d9[7]+d9[8]+d9[9])

################ Gráfica de punto maximo ó pico
plt.boxplot([D0, D1, D2, D3, D4, D5, D6, D7, D8, D9])
plt.scatter(1,(max(D0)), color="orange")
plt.scatter(2,(max(D1)), color="orange")
plt.scatter(3,(max(D2)), color="orange")
plt.scatter(4,(max(D3)), color="orange")
plt.scatter(5,(max(D4)), color="orange")
plt.scatter(6,(max(D5)), color="orange")
plt.scatter(7,(max(D6)), color="orange")
plt.scatter(8,(max(D7)), color="orange")
plt.scatter(9,(max(D8)), color="orange")
plt.scatter(10,(max(D9)), color="orange")
plt.plot([1,2,3,4,5,6,7,8,9,10], [(max(D0)),(max(D1)),(max(D2)),(max(D3)),(max(D4)),(max(D5)),(max(D6)),(max(D7)),(max(D8)),(max(D9))],color='orange', label='Máximos')
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ['0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9'])
plt.scatter(1,(np.median(D0)), color="red")
plt.scatter(2,(np.median(D1)), color="red")
plt.scatter(3,(np.median(D2)), color="red")
plt.scatter(4,(np.median(D3)), color="red")
plt.scatter(5,(np.median(D4)), color="red")
plt.scatter(6,(np.median(D5)), color="red")
plt.scatter(7,(np.median(D6)), color="red")
plt.scatter(8,(np.median(D7)), color="red")
plt.scatter(9,(np.median(D8)), color="red")
plt.scatter(10,(np.median(D9)), color="red")
plt.plot([1,2,3,4,5,6,7,8,9,10], [(np.median(D0)),(np.median(D1)),(np.median(D2)),(np.median(D3)),(np.median(D4)),(np.median(D5)),(np.median(D6)),(np.median(D7)),(np.median(D8)),(np.median(D9))],color='red', label='Medias')
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ['0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9'])                                                                                               
plt.xlabel('Probabilidad de vacunación')
plt.ylabel('Altura de pico')
plt.legend()
plt.legend(title='Cambio respecto al porcentaje')                                                                                                  
plt.show()


################ Gráfico de momento del pico

plt.boxplot([p0, p1, p2, p3, p4, p5, p6, p7, p8, p9])
plt.scatter(1,(max(p0)), color="orange")
plt.scatter(2,(max(p1)), color="orange")
plt.scatter(3,(max(p2)), color="orange")
plt.scatter(4,(max(p3)), color="orange")
plt.scatter(5,(max(p4)), color="orange")
plt.scatter(6,(max(p5)), color="orange")
plt.scatter(7,(max(p6)), color="orange")
plt.scatter(8,(max(p7)), color="orange")
plt.scatter(9,(max(p8)), color="orange")
plt.scatter(10,(max(p9)), color="orange")
plt.plot([1,2,3,4,5,6,7,8,9,10], [(max(p0)),(max(p1)),(max(p2)),(max(p3)),(max(p4)),(max(p5)),(max(p6)),(max(p7)),(max(p8)),(max(p9))],color='orange', label='Máximos')
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ['0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9'])
plt.scatter(1,(np.median(p0)), color="red")
plt.scatter(2,(np.median(p1)), color="red")
plt.scatter(3,(np.median(p2)), color="red")
plt.scatter(4,(np.median(p3)), color="red")
plt.scatter(5,(np.median(p4)), color="red")
plt.scatter(6,(np.median(p5)), color="red")
plt.scatter(7,(np.median(p6)), color="red")
plt.scatter(8,(np.median(p7)), color="red")
plt.scatter(9,(np.median(p8)), color="red")
plt.scatter(10,(np.median(p9)), color="red")
plt.plot([1,2,3,4,5,6,7,8,9,10], [(np.median(p0)),(np.median(p1)),(np.median(p2)),(np.median(p3)),(np.median(p4)),(np.median(p5)),(np.median(p6)),(np.median(p7)),(np.median(p8)),(np.median(p9))],color='red', label='Medias')
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ['0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9'])                                                                                               
plt.xlabel('Probabilidad de vacunación')
plt.ylabel('Momento del pico')
plt.legend()
plt.legend(title='Cambio respecto al porcentaje')                                                                                                  
plt.show()


    

