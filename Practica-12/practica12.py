from random import randint
from math import floor, log
import pandas as pd
import numpy as np
from pyDOE import lhs
import matplotlib.pyplot as plt
from cv2 import cv2
from tabulate import tabulate

########### genero posiciones dispersas en 3 planos
var=3
muestras =10
x= lhs(var, muestras, criterion="corr")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:,0], x[:,1], x[:,2])
plt.show()
print(x)
######### 
BOX=[]
MTP=[]
MFP=[]
MFN=[]
MTN=[]
pst=[]
for ciclos in range(0, muestras):

    posiciones= x[ciclos]
    posn=posiciones[0]
    posg=posiciones[1]
    posb=posiciones[2]

    pst.append(posiciones)
    print(posiciones,'posiciones n,g,b')
     
    modelos = pd.read_csv('digits.txt', sep=' ', header = None)
    modelos = modelos.replace({'n': posn, 'g': posg, 'b': posb})
    r, c = 5, 3
    dim = r * c
 
    tasa = 0.15
    tranqui = 0.99
    tope = 9
    k = tope + 1 # incl. cero
    contadores = np.zeros((k, k + 1), dtype = int)
    n = floor(log(k-1, 2)) + 1
    neuronas = np.random.rand(n, dim) # perceptrones
  
    for t in range(5000): # entrenamiento
        d = randint(0, tope)
        pixeles = 1 * (np.random.rand(dim) < modelos.iloc[d])
        correcto = '{0:04b}'.format(d)
        for i in range(n):
            w = neuronas[i, :]
            deseada = int(correcto[i]) # 0 o 1
            resultado = sum(w * pixeles) >= 0
            if deseada != resultado: 
                ajuste = tasa * (1 * deseada - 1 * resultado)
                tasa = tranqui * tasa 
                neuronas[i, :] = w + ajuste * pixeles
    prueba= 300
    for t in range(prueba): # prueba
        d = randint(0, tope)
        pixeles = 1 * (np.random.rand(dim) < modelos.iloc[d])
        correcto = '{0:04b}'.format(d)
        salida = ''
        for i in range(n):
            salida += '1' if sum(neuronas[i, :] * pixeles) >= 0 else '0'
        r = min(int(salida, 2), k)
        contadores[d, r] += 1
    
    contadores2=np.delete(contadores,10,1)# elimina la columna 10 para poder trabajar
    print(contadores2)
    TP= np.diag(contadores2)
    FP= np.sum(contadores2, axis=0)-TP
    FN = np.sum(contadores2, axis=1) - TP
    
    #print(TP, 'diagonal')
    #print(FP, 'columna')
    #print(FN, 'fila')

### columna de NA 10 11 12 13 14 15
    
    arreglo = np.array((contadores))
    NA= np.sum(arreglo[:, 10])
    
###
    num_classes = 10
    TN = []
    for i in range(num_classes):
        temp = np.delete(contadores2, i, 0)
        temp = np.delete(temp, i, 1)  
        TN.append(sum(sum(temp)))
    #print(TN,'TN')
        
    CNF = prueba
    for i in range(num_classes):
        print(TP[i] + FP[i] + FN[i] + TN[i] + NA == CNF)

    precision= TP/(TP+FP)

    MTP.append(sum(TP)/len(TP))
    MFP.append(sum(FP)/len(TP))
    MFN.append(sum(FN)/len(TP))
    MTN.append(sum(TN)/len(TP))


    
    recuperacion = TP/(TP+FN)
    especificidad = TN/(TN+FP)
    print(precision, 'precision')
    BOX.append(precision)
    #cv2.waitKey(20000)


plt.boxplot([BOX[0], BOX[1], BOX[2], BOX[3], BOX[4], BOX[5], BOX[6], BOX[7], BOX[8], BOX[9]])  
plt.xlabel('Probabilidades')
plt.ylabel('Resultados de precisión %')                                                                                                 
plt.show()

plt.boxplot([MTP, MFP, MFN])
plt.xticks([1, 2, 3], ['TP','FP','FN'])
plt.xlabel('Resultados')
plt.ylabel('Promedios')                                                                                                  
plt.show()

Tabla1 = [["muestra", "posiciones", "promedio", "promedio"],
          ["1", pst[0], (sum(BOX[0])/10), MTP[0]],
          ["1", pst[1], (sum(BOX[1])/10), MTP[1]],
          ["1", pst[2], (sum(BOX[2])/10), MTP[2]],
          ["1", pst[3], (sum(BOX[3])/10), MTP[3]],
          ["1", pst[4], (sum(BOX[4])/10), MTP[4]],
          ["1", pst[5], (sum(BOX[5])/10), MTP[5]],
          ["1", pst[6], (sum(BOX[6])/10), MTP[6]],
          ["1", pst[7], (sum(BOX[7])/10), MTP[7]],
          ["1", pst[8], (sum(BOX[8])/10), MTP[8]],
          ["1", pst[9], (sum(BOX[9])/10), MTP[9]]]
print(tabulate(Tabla1))







#print(recuperacion, 'recuperacion')
#print(especificidad, 'especificidad')

###### imprimir visual nada mas 
#c = pd.DataFrame(contadores)
#c.columns = [str(i) for i in range(k)] + ['NA']
#c.index = [str(i) for i in range(k)]
#print(c)
