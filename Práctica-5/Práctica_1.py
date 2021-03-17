from GeneralRandom import GeneralRandom
from math import exp, pi
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import pylab

def g(x):
    return (2  / (pi * (exp(x) + exp(-x))))
 
vg = np.vectorize(g)
X = np.arange(-8, 8, 0.05) # ampliar y refinar
Y = vg(X) # mayor eficiencia
 
desde = 2.96
hasta = 7
replicas = 100
digito1=[]
digito2=[]
comparacion=[]
similar=[]
valoresG = []
dos=[]
tres=[]
cuatro=[]

for decimales in (2, 3, 4):
    
    if decimales == 2:
        n=81000#90%
    if decimales == 3:
        n=530000#80%
    if decimales == 4:
        n=880000#10%
    for C in range(replicas):
        generador = GeneralRandom(np.asarray(X), np.asarray(Y))
        V = generador.random(n)[0]
        mc = ((V >= desde) & (V <= hasta))
        integral = sum(mc) / n
        numero=((pi / 2) * integral)
        #print(numero)
            
        valor1= 0.048834
        valor2= numero

        if decimales== 2:
            dos.append(valor2)
        if decimales== 3:
            tres.append(valor2)
        if decimales== 4:
            cuatro.append(valor2)

        
        for x in str(valor1):
            digito1.append(x)
        for y in str(valor2):
            digito2.append(y)
        for z in range(decimales):
            A=(int(digito1[z+2])==int(digito2[z+2]))#paso entero los string
            comparacion.append(A)    
        if all(s==True for s in comparacion)== False:
            similar.append(1)
            
        digito1.clear()
        digito2.clear()
        comparacion.clear()
    cuantos=(len(similar))
    porcentaje = (((replicas-cuantos)*100)/(replicas))
    print(porcentaje,'%')
    valoresG.append(porcentaje)
    similar.clear()

H=[2]
V=(81000)
H2=[3]
V2=(530000)
H3=[4]
V3=(880000)

plt.plot([H, H2, H3], [V, V2, V3])
plt.scatter(H,V, color="orange", label='90%')
plt.scatter(H2,V2, color="red", label='80%')
plt.scatter(H3,V3, color="green", label='10%')
plt.legend()
plt.legend(title='Porcentaje de efectividad')
plt.yticks([0,81000,200000,300000,400000,500000,600000,700000,800000,900000,1000000], ['0','81000','200000','300000','400000','500000','600000','700000','800000','900000','1000000'])
plt.xticks([2,3,4])
plt.xlabel('Decimales')
plt.ylabel('Muestras')
plt.savefig('p5pf.png')
plt.show() # opcional


plt.boxplot([dos, tres, cuatro])
plt.xticks([1,2,3],['2','3','4'])
plt.xlabel('Decimales')
plt.ylabel('Valor Pseudoaleatoria')
plt.show()
plt.close()












