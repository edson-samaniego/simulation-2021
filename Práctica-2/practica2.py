import numpy 
from random import randint
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import pyplot
import cv2 as cv2

z=[]
fila = 12
col = 12
matriz = [[randint(0,1) for i in range(fila)]for j in range(col)]
matriz2=[]
matriz3=[]
matrizinicial=[]
rep=30
iteraciones=[]
f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12=[],[],[],[],[],[],[],[],[],[],[],[]
f1i,f2i,f3i,f4i,f5i,f6i,f7i,f8i,f9i,f10i,f11i,f12i=[],[],[],[],[],[],[],[],[],[],[],[]

#se crea la matriz de 12x12 visual
for x in range(fila):
    for y in range(col):
        print(matriz[x][y],end=' ')
        matriz2.append(matriz[x][y])
        matriz3.append(matriz[x][y])        
    print()
    
# este for cambiara la posicion desde 0 a 143
for i in range(rep):
    for z in range (0, 144):
        pos=matriz2[z]
    
        v0=z-13 
        v1=z-12
        v2=z-11
        v3=z-1
        v5=z+1
        v6=z+11
        v7=z+12
        v8=z+13
        if 130>=z >=0:
            L=(matriz2[v0]+matriz2[v1]+matriz2[v2]+matriz2[v3]+matriz2[v5]+matriz2[v6]+matriz2[v7]+matriz2[v8])    
            if not z%12:
                L=(matriz2[v1]+matriz2[v2]+matriz2[v5]+matriz2[v7]+matriz2[v8])
                if z == 0:
                    L=(matriz2[v5]+matriz2[v7]+matriz2[v8])
            if not (z+1)%12:
                L=(matriz2[v0]+matriz2[v1]+matriz2[v3]+matriz2[v6]+matriz2[v7])
                if z == 11:
                    L=(matriz2[v3]+matriz2[v6]+matriz2[v7])
            if 0< z <11: 
                L=(matriz2[v3]+matriz2[v5]+matriz2[v6]+matriz2[v7]+matriz2[v8])      
        if z == 131:
            L=(matriz2[v0]+matriz2[v1]+matriz2[v3]+matriz2[v6]+matriz2[v7])
        if z == 132:
            L=(matriz2[v1]+matriz2[v2]+matriz2[v5])
        if 143> z >132:
            L=(matriz2[v0]+matriz2[v1]+matriz2[v2]+matriz2[v3]+matriz2[v5])
        if z == 143:
            L=(matriz2[v0]+matriz2[v1]+matriz2[v3])

        if pos == 1:  
            if L < 3 : 
                matriz3[z]= 0           
        if pos == 0:
            if 7 > L > 4
                 matriz3[z]= 1 
            
                
            
                
######################################
    matrizinicial=matriz2
    matriz2=matriz3 #hago que la matriz revisada tome el valor de la matriz modificada
    iteraciones.append(sum(matriz3))# solo para ver cuantos vivos hay 
    print(i)#imprimo la iteracion que va
    
    for n in range(0,12):
        f1.append(matriz2[n])
        f1i.append(matrizinicial[n])
    for n in range(12,24):
        f2.append(matriz2[n])
        f2i.append(matrizinicial[n])
    for n in range(24,36):
        f3.append(matriz2[n])
        f3i.append(matrizinicial[n])
    for n in range(36,48):
        f4.append(matriz2[n])
        f4i.append(matrizinicial[n])
    for n in range(48,60):
        f5.append(matriz2[n])
        f5i.append(matrizinicial[n])
    for n in range(60,72):
        f6.append(matriz2[n])
        f6i.append(matrizinicial[n])
    for n in range(72,84):
        f7.append(matriz2[n])
        f7i.append(matrizinicial[n])
    for n in range(84,96):
        f8.append(matriz2[n])
        f8i.append(matrizinicial[n])
    for n in range(96,108):
        f9.append(matriz2[n])
        f9i.append(matrizinicial[n])
    for n in range(108,120):
        f10.append(matriz2[n])
        f10i.append(matrizinicial[n])
    for n in range(120,132):
        f11.append(matriz2[n])
        f11i.append(matrizinicial[n])
    for n in range(132,144):
        f12.append(matriz2[n])
        f12i.append(matrizinicial[n])
        
    M = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12]
    M2 = [f1i,f2i,f3i,f4i,f5i,f6i,f7i,f8i,f9i,f10i,f11i,f12i]
    print(M)
    matrix = numpy.matrix(M)   
    fig = plt.figure()
    plt.xlabel('12 columnas')
    plt.ylabel('12 filas')
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Greys)
    fig.suptitle(i + 1)
    plt.savefig('regla5_it1.png')
    plt.show()
    
    if i == 0:
        matrix = numpy.matrix(M2)   
        fig = plt.figure()
        plt.xlabel('12 columnas')
        plt.ylabel('12 filas')
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Greys)
        fig.suptitle('inicio')
        plt.savefig('regla5_it1.png')
        plt.show()        
        
    f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12=[],[],[],[],[],[],[],[],[],[],[],[]
    

pasos=["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30"]
valores= iteraciones
plt.figure(figsize=(20,10))
pyplot.title("grafica de celdas vivas por iteracion", fontsize=20)
pyplot.bar(pasos, height = valores, width=0.5)
pyplot.ylabel("celdas vivas", fontsize=20)
pyplot.xlabel("iteraciones", fontsize=20)
pyplot.savefig("grafica_barra.png")
pyplot.show()

print(iteraciones)


    

        

       
