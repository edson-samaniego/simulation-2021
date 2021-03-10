import seaborn as sns
from math import sqrt, fabs
from scipy.stats import describe
from PIL import Image, ImageColor
from random import randint, choice
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from cv2 import cv2
##################
vectorX=[]
vectorY=[]
coordenada=[]
Eucl=[]
Manh=[]
borde=[]
Manh2= []
total=[]
##################
n, semillas = 40, []
for SEM in (5, 30, 100):
    k = SEM
    for s in range(k):
        while True:
            x, y = randint(0, n - 1), randint(0, n - 1)
            if (x, y) not in semillas:
                semillas.append((x, y))
                break
 
    def celda(pos):
        if pos in semillas:
            return semillas.index(pos)
        x, y = pos % n, pos // n
        cercano = None
        menor = n * sqrt(2)
        for i in range(k):
            (xs, ys) = semillas[i]
            dx, dy = x - xs, y - ys
            dist = sqrt(dx**2 + dy**2)
            if dist < menor:
                cercano, menor = i, dist
        return cercano
 
    def inicio():
        direccion = randint(0, 3)
        if direccion == 0: # vertical abajo -> arriba
            return (0, randint(0, n - 1))
        elif direccion == 1: # izq. -> der
            return (randint(0, n - 1), 0)
        elif direccion == 2: # der. -> izq.
            return (randint(0, n - 1), n - 1)
        else:
            return (n - 1, randint(0, n - 1))
 
    celdas = [celda(i) for i in range(n * n)]
    voronoi = Image.new('RGB', (n, n))
    vor = voronoi.load()
    c = sns.color_palette("Set3", k).as_hex()
    for i in range(n * n):
        vor[i % n, i // n] = ImageColor.getrgb(c[celdas.pop(0)])
    limite, vecinos = n, []
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            if dx != 0 or dy != 0:
                vecinos.append((dx, dy))
 
    def propaga(replica):
        prob, dificil = 0.9, 0.8
        grieta = voronoi.copy()
        g = grieta.load()
        (x, y) = inicio()
        largo = 0
        negro = (0, 0, 0)
        while True:
            g[x, y] = negro
            largo += 1
            frontera, interior = [], []
            for v in vecinos:
                (dx, dy) = v
                vx, vy = x + dx, y + dy
                if vx >= 0 and vx < n and vy >= 0 and vy < n: # existe
                   if g[vx, vy] != negro: # no tiene grieta por el momento
                       if vor[vx, vy] == vor[x, y]: # misma celda
                           interior.append(v)
                       else:
                           frontera.append(v)
            elegido = None
            if len(frontera) > 0:
                elegido = choice(frontera)
                prob = 1
            elif len(interior) > 0:
                elegido = choice(interior)
                prob *= dificil
            if elegido is not None:
                (dx, dy) = elegido
                x, y = x + dx, y + dy
            else:
                break # ya no se propaga
    
####################################       
        for R in range(0,n):
            for S in range(0,n):
                columna1 = R
                fila1= S
                if g[columna1,fila1] == (0, 0, 0):
                    vectorX.append(R)
                    vectorY.append(S)
                    coordenada.append([R,S])
        total=(len(coordenada))# total de píxeles de grieta 
        print(total)          
        for T in range(0, total):
            p=coordenada[T] 
            XS=p[0]         # obtengo el X de la grieta 
            YS=p[1]         # obtengo el Y de la grieta 
            origen = [((n-1)/2), ((n-1)/2)]
            OX = origen[0]
            OY = origen[1]
            DE= sqrt((XS-OX)**2+(YS-OY)**2) 
            Eucl.append((DE,coordenada[T])) 
            c=(min(Eucl))   # obtengo el píxel más cercano al centro 
            if XS== (n-1) or YS==(n-1):
                borde.append(coordenada[T])
            if XS==0 or YS==0:
                borde.append(coordenada[T])
        cercano=(c[1])      
        FX= cercano[0]
        FY= cercano[1]
        lejano=borde[0]
        IX=lejano[0]
        IY=lejano[1]
        DMx= fabs(FX-IX)
        DMy= fabs(FY-IY)
        DM = DMx + DMy       
        Manh.append(DM)
        Manh2.append(DM) 
#####################################
        if largo >= limite:
            visual = grieta.resize((10 * n,10 * n))
            visual.save("p4pg_{:d}.png".format(replica))
        return largo    
    for r in range(20): 
        borde.clear()
        coordenada.clear()
        Eucl.clear()
        Manh.clear()
        vectorX.clear()
        vectorY.clear()
        propaga(r)
###################################
for H in range (len(Manh2)):
    suma= Manh2[H]
    total.append(suma)
print(total)
pocas= total[0:20]
regular= total[20:40]
muchos= total[40:60]


plt.boxplot([pocas, regular, muchos])
plt.xlabel('              5 semillas                     30 semillas                  100 semillas')
plt.ylabel('Distancia máxima Manhattan')
plt.show()
plt.close()


min1=(min(pocas))
max1=(max(pocas))
pro1=(np.mean(pocas))
med1=(np.median(pocas))

min2=(min(regular))
max2=(max(regular))
pro2=(np.mean(regular))
med2=(np.median(regular))

min3=(min(muchos))
max3=(max(muchos))
pro3=(np.mean(muchos))
med3=(np.median(muchos))

Tabla1 = [["semillas", "minimo", "maximo", "promedio", "media"],
          ["5", min1, max1, pro1, med1],
          ["30", min2, max2, pro2, med2],
          ["100", min3, max3, pro3, med3]]
print(tabulate(Tabla1))
















