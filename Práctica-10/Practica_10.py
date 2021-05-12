from time import time
import numpy as np
import pandas as pd
from random import random, randint, sample
from cv2 import cv2
import math
 
def knapsack(peso_permitido, pesos, valores):
    assert len(pesos) == len(valores)
    peso_total = sum(pesos)
    valor_total = sum(valores)
    if peso_total < peso_permitido: 
        return valor_total
    else:
        V = dict()
        for w in range(peso_permitido + 1):
            V[(w, 0)] = 0
        for i in range(len(pesos)):
            peso = pesos[i]
            valor = valores[i]
            for w in range(peso_permitido + 1):
                cand = V.get((w - peso, i), -float('inf')) + valor
                V[(w, i + 1)] = max(V[(w, i)], cand)
        return max(V.values())

def generador_pesos(cuantos, low, high):
    return np.round(normalizar(np.random.normal(size = cuantos)) * (high - low) + low)
 
def generador_valores(pesos, low, high):
    n = len(pesos)
    valores = np.empty((n))
    for i in range(n):
        valores[i] = np.random.normal(pesos[i], random(), 1)
    return normalizar(valores) * (high - low) + low

def factible(seleccion, pesos, capacidad):
    return np.inner(seleccion, pesos) <= capacidad
  
def objetivo(seleccion, valores):
    return np.inner(seleccion, valores)
 
def normalizar(data):
    menor = min(data)
    mayor = max(data)
    rango  = mayor - menor
    data = data - menor # > 0
    return data / rango # entre 0 y 1
  

 
def poblacion_inicial(n, tam):
    pobl = np.zeros((tam, n))
    for i in range(tam):
        pobl[i] = (np.round(np.random.uniform(size = n))).astype(int)
    return pobl
 
def mutacion(sol, n):
    pos = randint(0, n - 1)
    mut = np.copy(sol)
    mut[pos] = 1 if sol[pos] == 0 else 0
    return mut
  
def reproduccion(x, y, n):
    pos = randint(2, n - 2)
    xy = np.concatenate([x[:pos], y[pos:]])
    yx = np.concatenate([y[:pos], x[pos:]])
    return (xy, yx)

def ruleta(lista,n):
    import random
    F, NF = [],[]
    f=len(lista.loc[d.fact == True,])
    nf=len(lista.loc[d.fact == False,])
    for i in range(0,f):
        F.append(i)
    for j in range(f,(f+nf)):
        NF.append(j)

    if len(NF)==0:
        for k in range(0,3):
            NF.append(k)
    listas=[(random.sample(F, k=2)),(random.sample(NF, k=2))]
    results= random.choices(listas, weights= [len(F), len(NF)], k = 2)
    if results[0]==results[1]:
        salida= results[0]
    if results[0]!=results[1]:
        dato1= results[0]
        dato2= results[1]
        salida= [dato1[0],dato2[0]]
    return(salida)

mejores=[]
best=[]
bestR1=[]
bestR2=[]
bestR3=[]
tiempociclo=[]
for regla in (1, 2, 3):
    pesosexp=[]
    diferentes=[]
    vdif=[]
    vd=[]

    ANTES= time()
    
    #### independientes 1
    if regla == 1:
        n=50
        pesos = generador_pesos(n, 15, 80)
        for B in range(0,len(pesos)):
            diferentes.append(randint(0,300))
        valores = generador_valores(diferentes, 10, 500)
    #### distribucion exponencial
    if regla == 2:
        n=350
        for C in range(0,n):
            vdif.append(randint(0,20))
        for D in range(0, n):
            vd.append(math.exp(vdif[D]))
        pesos = generador_pesos(n, 15, 80)
        valores = generador_valores(vd, 10, 500)
    #### al cuadrado
    if regla == 3:
        n=60
        pesos = generador_pesos(n, 15, 80)# genera 50 aleatorio en rango de 15 a 80
        for A in range(0,len(pesos)):
            pesosexp.append(pesos[A]**2)
        valores = generador_valores(pesosexp, 10, 500)#50 valores entre 10 a 500

    capacidad = int(round(sum(pesos) * 0.65))#da la capacidad maxima de los pesos
    optimo = knapsack(capacidad, pesos, valores)#dato optimo de capacidad de pesos
    best.append(optimo)
    
    for ciclo in range(0,2):
        init = 200
        p = poblacion_inicial(n, init)
        tam = p.shape[0]
        assert tam == init
        pm = 0.05
        rep = 50
        tmax = 50
        mejor = None
        mejoresruleta=[]
        for t in range(tmax):
            d = []
            for i in range(tam):
                d.append({'idx': i, 'obj': objetivo(p[i], valores),
                          'fact': factible(p[i], pesos, capacidad)})

            d = pd.DataFrame(d).sort_values(by = ['fact', 'obj'], ascending = False)
            listaR=pd.DataFrame(d).sort_values(by=['idx'], ascending=0)
    
            for i in range(tam): # mutarse con probabilidad pm
                if random() < pm:
                    p = np.vstack([p, mutacion(p[i], n)])
                
            for i in range(rep):  # reproducciones
                if ciclo == 0:
                    padres = sample(range(tam), 2)
                if ciclo == 1:
                    padres= ruleta(d,tam)
                hijos = reproduccion(p[padres[0]], p[padres[1]], n)
                p = np.vstack([p, hijos[0], hijos[1]])

            tam = p.shape[0]
            d = []
            for i in range(tam):
                d.append({'idx': i, 'obj': objetivo(p[i], valores),
                          'fact': factible(p[i], pesos, capacidad)})

            d = pd.DataFrame(d).sort_values(by = ['fact', 'obj'], ascending = False)
            mantener = np.array(d.idx[:init])
            p = p[mantener, :]
            tam = p.shape[0]
            assert tam == init
            factibles = d.loc[d.fact == True,]
            infactibles = d.loc[d.fact == False,]
            mejor = max(factibles.obj)
            mejores.append(mejor)
        #print(mejores)
        #print(len(mejores),'largo de cuantos mejores')
    if regla == 1:
        bestR1.append(max(mejores[0:50]))
        if ciclo == 1:
            bestR1.append(max(mejores[50:100]))
    if regla == 2:    
        bestR2.append(max(mejores[100:150]))
        if ciclo == 1:
            bestR2.append(max(mejores[150:200]))
    if regla == 3:
        bestR3.append(max(mejores[200:250]))
        if ciclo == 1:
            bestR3.append(max(mejores[250:300]))
    tiempociclo.append(time() - ANTES)
##################################
print(bestR1, 'mejores mas cercano a la franja verde')
print(bestR2, 'mejores mas cercano a la franja verde')
print(bestR3, 'mejores mas cercano a la franja verde')
print(best,'optimo ')
print(tiempociclo, 'tiempo por gráfico')
#cv2.waitKey(10000)
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 3), dpi=300)
plt.plot(range(tmax), mejores[0:50], linewidth=1, color='blue',label='Sin ruleta')
plt.plot(range(tmax), mejores[50:100], linewidth=1, color='red',label='Con ruleta')
plt.scatter(range(tmax), mejores[0:50], color='blue', s=10)
plt.scatter(range(tmax), mejores[50:100], color='red', s=10)
plt.axhline(y = (best[0]), color = 'green', linewidth=3)
plt.xlabel('Paso')
plt.ylabel('Mayor valor')
plt.legend()
plt.legend(loc='lower right', title='Mejores resultados')
plt.savefig('p10p.png', bbox_inches='tight')
#plt.show()
plt.close()

plt.figure(figsize=(7, 3), dpi=300)
plt.plot(range(tmax), mejores[100:150], linewidth=1, color='blue',label='Sin ruleta')
plt.plot(range(tmax), mejores[150:200], linewidth=1, color='red',label='Con ruleta')
plt.scatter(range(tmax), mejores[100:150], color='blue', s=10)
plt.scatter(range(tmax), mejores[150:200], color='red', s=10)
plt.axhline(y = (best[1]), color = 'green', linewidth=3)
plt.xlabel('Paso')
plt.ylabel('Mayor valor')
plt.legend()
plt.legend(loc='lower right', title='Mejores resultados')
plt.savefig('p11p.png', bbox_inches='tight')
#plt.show()
plt.close()

plt.figure(figsize=(7, 3), dpi=300)
plt.plot(range(tmax), mejores[200:250], linewidth=1, color='blue',label='Sin ruleta')
plt.plot(range(tmax), mejores[250:300], linewidth=1, color='red',label='Con ruleta')
plt.scatter(range(tmax), mejores[200:250], color='blue', s=10)
plt.scatter(range(tmax), mejores[250:300], color='red', s=10)
plt.axhline(y = (best[2]), color = 'green', linewidth=3)
plt.xlabel('Paso')
plt.ylabel('Mayor valor')
plt.legend()
plt.legend(loc='lower right', title='Mejores resultados')
plt.savefig('p12p.png', bbox_inches='tight')
#plt.show()
plt.close()


#print(mejor, (optimo - mejor) / optimo)
#print(mejores)
#print(sorted(mejores))
