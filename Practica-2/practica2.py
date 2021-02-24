import numpy as np # hay que instalar numpy a parte con pip3 o algo similar
from random import randint
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import cv2 as cv2
z=[]

fila = 12
col = 12
matriz = [[randint(0,1) for i in range(fila)]for j in range(col)]
matriz2=[]
#se crea la matriz de 12x12 visual
for x in range(fila):
    for y in range(col):
        print(matriz[x][y],end=' ')
        matriz2.append(matriz[x][y])
    print()
#mando los datos de la matriz a un vector de 0 a 143    
print(matriz2)

# este for cambiara la posicion desde 0 a 143
for z in range (0, 144):
    v0=z-13
    v1=z-12
    v2=z-11
    v3=z-1
    v5=z+1    
    v6=z+11
    v7=z+12
    v8=z+13
    pos=matriz2[z]
    print(pos,'centro',z) #imprimo para ver en que posicion va
    print(matriz2[v1], 'vecino')   #visualizo el vecino que le indico 
    if pos == 1:
        if matriz2[v1] == 0:##### regla 1 si el vecino superior es 0 
            
            matriz2[z]=0 #### el centro cambiara a muerto osea de 1 a 0
            
            print(matriz2[z])
            cv2.waitKey(5000000)
        
        cv2.waitKey(10)
    
    
    
    

        

       
