import numpy as np
import pandas as pd
from math import sqrt, fabs
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
import matplotlib.colorbar as colorbar
from matplotlib.colors import LinearSegmentedColormap
from cv2 import cv2

paso = 256 // 10
niveles = [i/256 for i in range(0, 256, paso)]
colores = [(niveles[i], niveles[-(i + 1)], 0) for i in range(len(niveles))]
palette = LinearSegmentedColormap.from_list('tonos', colores, N = len(colores))
 
from math import fabs, sqrt, floor, log
eps = 0.001
 
### esta funcion calcula la fuerza de la particula i segun la posicion dada
def fuerza(i, shared):
    p = shared.data
    n = shared.count
    pi = p.iloc[i]
    xi = pi.x
    yi = pi.y
    ci = pi.c
    mi = pi.m

    factorm = 1
    factorc = 1
    fx, fy = 0, 0
    for j in range(n): #este ciclo revisa todas las particulas 
        pj = p.iloc[j]
        cj = pj.c
        dire = (-1)**(1 + (ci * cj < 0))
        dx = xi - pj.x
        dy = yi - pj.y
        factor = dire * fabs(ci - cj) / (sqrt(dx**2 + dy**2) + eps)
        fx -= dx * factor * factorc
        fy -= dy * factor * factorc

    for j in range(n): # ciclo que afecta respecto a la masa 
        pj = p.iloc[j]
        mj = pj.m
        dire = 2
        dx = xi - pj.x
        dy = yi - pj.y
        factor = dire * (mj - mi) / (sqrt(dx**2 + dy**2) + eps)
        fx -= dx * factor * factorm
        fy -= dy * factor * factorm
            
    return (fx, fy)
 
from os import popen
 
def actualiza(pos, fuerza, de):
    return max(min(pos + de * fuerza, 1), 0)

import multiprocessing
from itertools import repeat

if __name__ == "__main__":
    popen('rm -f p9p_t*.png') # borramos anteriores en el caso que lo hayamos corrido
  
    n = 20
    x = np.random.normal(size = n)
    y = np.random.normal(size = n)
    c = np.random.normal(size = n)
    m = np.random.normal(size = n)

    Min = 0
    Max = 10
    mmax = max(m)
    mmin = min(m)
    m = Max * (m - mmin) / (mmax - mmin) + Min

    
    xmax = max(x)
    xmin = min(x)
    x = (x - xmin) / (xmax - xmin) # de 0 a 1
    ymax = max(y)
    ymin = min(y)
    y = (y - ymin) / (ymax - ymin) 
    cmax = max(c)
    cmin = min(c)
    c = 2 * (c - cmin) / (cmax - cmin) - 1 # entre -1 y 1 diferente valor de fuerza de carga 
    g = np.round(5 * c).astype(int)
    p = pd.DataFrame({'x': x, 'y': y, 'c': c, 'g': g, 'm': m})
    mgr = multiprocessing.Manager() # https://stackoverflow.com/questions/22487296/multiprocessing-in-python-sharing-large-object-e-g-pandas-dataframe-between
    ns = mgr.Namespace()
    ns.data = p # compartido entre el pool
    ns.count = n

    cicl=[]
    DEUC0=[]
    DEUC1=[]
    DEUC2=[]
    DEUC3=[]
    DEUC4=[]
    DEUC5=[]
    DEUC6=[]
    DEUC7=[]
    DEUC8=[]
    DEUC9=[]
    DEUC10=[]
    DEUC11=[]
    DEUC12=[]
    DEUC13=[]
    DEUC14=[]
    DEUC15=[]
    DEUC16=[]
    DEUC17=[]
    DEUC18=[]
    DEUC19=[]
    p1x, p1y = [], []
    p2x, p2y = [], []
    p3x, p3y = [], []
    p4x, p4y = [], []
    p5x, p5y = [], []
    p6x, p6y = [], []
    p7x, p7y = [], []
    p8x, p8y = [], []
    p9x, p9y = [], []
    p10x, p10y = [], []
    p11x, p11y = [], []
    p12x, p12y = [], []
    p13x, p13y = [], []
    p14x, p14y = [], []
    p15x, p15y = [], []
    p16x, p16y = [], []
    p17x, p17y = [], []
    p18x, p18y = [], []
    p19x, p19y = [], []
    p20x, p20y = [], []


    
    tmax = 20
    digitos = floor(log(tmax, 10)) + 1
    fig, ax = plt.subplots(figsize=(6, 5), ncols=1)
    pos = plt.scatter(p.x, p.y, c = p.g, s = round(p.m*10), marker = 's', cmap = palette)
    fig.colorbar(pos, ax=ax)
    plt.title('Estado inicial')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    fig.savefig('p9p_t0.png')
    plt.close()
    print(p)
    xin0= p.x[0]
    yin0= p.x[0]
    xin1= p.x[1]
    yin1= p.x[1]
    xin2= p.x[2]
    yin2= p.x[2]
    xin3= p.x[3]
    yin3= p.x[3]
    xin4= p.x[4]
    yin4= p.x[4]
    xin5= p.x[5]
    yin5= p.x[5]
    xin6= p.x[6]
    yin6= p.x[6]
    xin7= p.x[7]
    yin7= p.x[7]
    xin8= p.x[8]
    yin8= p.x[8]
    xin9= p.x[9]
    yin9= p.x[9]
    xin10= p.x[10]
    yin10= p.x[10]
    xin11= p.x[11]
    yin11= p.x[11]
    xin12= p.x[12]
    yin12= p.x[12]
    xin13= p.x[13]
    yin13= p.x[13]
    xin14= p.x[14]
    yin14= p.x[14]
    xin15= p.x[15]
    yin15= p.x[15]
    xin16= p.x[16]
    yin16= p.x[16]
    xin17= p.x[17]
    yin17= p.x[17]
    xin18= p.x[18]
    yin18= p.x[18]
    xin19= p.x[19]
    yin19= p.x[19]
    for t in range(tmax):
        with multiprocessing.Pool() as pool: # rehacer para que vea cambios en p
            f = pool.starmap(fuerza, [(i, ns) for i in range(n)])
            delta = 0.02 / max([max(fabs(fx), fabs(fy)) for (fx, fy) in f])
            p['x'] = pool.starmap(actualiza, zip(p.x, [v[0] for v in f], repeat(delta)))
            p['y'] = pool.starmap(actualiza, zip(p.y, [v[1] for v in f], repeat(delta)))
            fig, ax = plt.subplots(figsize=(6, 5), ncols=1)
            pos = plt.scatter(p.x, p.y, c = p.g, s = p.m*10, marker = 's', cmap = palette)
            fig.colorbar(pos, ax=ax)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.xlim(-0.1, 1.1)
            plt.ylim(-0.1, 1.1)            
            plt.title('Paso {:d}'.format(t + 1))
            fig.savefig('p9p_t' + format(t + 1, '0{:d}'.format(digitos)) + '.png')
            plt.close()
        cicl.append(t)
        p1x.append(p.x[0])
        p1x.append(p.x[1])
        p1x.append(p.x[2])
        p1x.append(p.x[3])
        p1x.append(p.x[4])
        p1x.append(p.x[5])
        p1x.append(p.x[6])
        p1x.append(p.x[7])
        p1x.append(p.x[8])
        p1x.append(p.x[9])
        p1x.append(p.x[10])
        p1x.append(p.x[11])
        p1x.append(p.x[12])
        p1x.append(p.x[13])
        p1x.append(p.x[14])
        p1x.append(p.x[15])
        p1x.append(p.x[16])
        p1x.append(p.x[17])
        p1x.append(p.x[18])
        p1x.append(p.x[19])

        p1y.append(p.y[0])
        p1y.append(p.y[1])
        p1y.append(p.y[2])
        p1y.append(p.y[3])
        p1y.append(p.y[4])
        p1y.append(p.y[5])
        p1y.append(p.y[6])
        p1y.append(p.y[7])
        p1y.append(p.y[8])
        p1y.append(p.y[9])
        p1y.append(p.y[10])
        p1y.append(p.y[11])
        p1y.append(p.y[12])
        p1y.append(p.y[13])
        p1y.append(p.y[14])
        p1y.append(p.y[15])
        p1y.append(p.y[16])
        p1y.append(p.y[17])
        p1y.append(p.y[18])
        p1y.append(p.y[19])
        
    for L in range(0, tmax):
        
        if L == 0:
            DEUC0.append(sqrt((p1x[0]-xin0)**2+(p1y[0]-yin0)**2))
            DEUC1.append(sqrt((p1x[1]-xin1)**2+(p1y[1]-yin1)**2))
            DEUC2.append(sqrt((p1x[2]-xin2)**2+(p1y[2]-yin2)**2))
            DEUC3.append(sqrt((p1x[3]-xin3)**2+(p1y[3]-yin3)**2))
            DEUC4.append(sqrt((p1x[4]-xin4)**2+(p1y[4]-yin4)**2))
            DEUC5.append(sqrt((p1x[5]-xin5)**2+(p1y[5]-yin5)**2))
            DEUC6.append(sqrt((p1x[6]-xin6)**2+(p1y[6]-yin6)**2))
            DEUC7.append(sqrt((p1x[7]-xin7)**2+(p1y[7]-yin7)**2))
            DEUC8.append(sqrt((p1x[8]-xin8)**2+(p1y[8]-yin8)**2))
            DEUC9.append(sqrt((p1x[9]-xin9)**2+(p1y[9]-yin9)**2))
            DEUC10.append(sqrt((p1x[10]-xin10)**2+(p1y[10]-yin10)**2))
            DEUC11.append(sqrt((p1x[11]-xin11)**2+(p1y[11]-yin11)**2))
            DEUC12.append(sqrt((p1x[12]-xin12)**2+(p1y[12]-yin12)**2))
            DEUC13.append(sqrt((p1x[13]-xin13)**2+(p1y[13]-yin13)**2))
            DEUC14.append(sqrt((p1x[14]-xin14)**2+(p1y[14]-yin14)**2))
            DEUC15.append(sqrt((p1x[15]-xin15)**2+(p1y[15]-yin15)**2))
            DEUC16.append(sqrt((p1x[16]-xin16)**2+(p1y[16]-yin16)**2))
            DEUC17.append(sqrt((p1x[17]-xin17)**2+(p1y[17]-yin17)**2))
            DEUC18.append(sqrt((p1x[18]-xin18)**2+(p1y[18]-yin18)**2))
            DEUC19.append(sqrt((p1x[19]-xin19)**2+(p1y[19]-yin19)**2))
            
        if L > 0:            
            DEUC1.append(sqrt((p1x[1]-p1x[L-1])**2+(p1y[1]-p1y[L-1])**2))
            DEUC2.append(sqrt((p1x[2]-p1x[L-1])**2+(p1y[2]-p1y[L-1])**2))
            DEUC3.append(sqrt((p1x[3]-p1x[L-1])**2+(p1y[3]-p1y[L-1])**2))
            DEUC4.append(sqrt((p1x[4]-p1x[L-1])**2+(p1y[4]-p1y[L-1])**2))
            DEUC5.append(sqrt((p1x[5]-p1x[L-1])**2+(p1y[5]-p1y[L-1])**2))
            DEUC6.append(sqrt((p1x[6]-p1x[L-1])**2+(p1y[6]-p1y[L-1])**2))
            DEUC7.append(sqrt((p1x[7]-p1x[L-1])**2+(p1y[7]-p1y[L-1])**2))
            DEUC8.append(sqrt((p1x[8]-p1x[L-1])**2+(p1y[8]-p1y[L-1])**2))
            DEUC9.append(sqrt((p1x[9]-p1x[L-1])**2+(p1y[9]-p1y[L-1])**2))
            DEUC10.append(sqrt((p1x[10]-p1x[L-1])**2+(p1y[10]-p1y[L-1])**2))
            DEUC11.append(sqrt((p1x[11]-p1x[L-1])**2+(p1y[11]-p1y[L-1])**2))
            DEUC12.append(sqrt((p1x[12]-p1x[L-1])**2+(p1y[12]-p1y[L-1])**2))
            DEUC13.append(sqrt((p1x[13]-p1x[L-1])**2+(p1y[13]-p1y[L-1])**2))
            DEUC14.append(sqrt((p1x[14]-p1x[L-1])**2+(p1y[14]-p1y[L-1])**2))
            DEUC15.append(sqrt((p1x[15]-p1x[L-1])**2+(p1y[15]-p1y[L-1])**2))
            DEUC16.append(sqrt((p1x[16]-p1x[L-1])**2+(p1y[16]-p1y[L-1])**2))
            DEUC17.append(sqrt((p1x[17]-p1x[L-1])**2+(p1y[17]-p1y[L-1])**2))
            DEUC18.append(sqrt((p1x[18]-p1x[L-1])**2+(p1y[18]-p1y[L-1])**2))
            DEUC19.append(sqrt((p1x[19]-p1x[L-1])**2+(p1y[19]-p1y[L-1])**2))
    print((sum(DEUC0))/(len(DEUC0)))
    print((sum(DEUC1))/(len(DEUC1)))
    print((sum(DEUC2))/(len(DEUC2)))
    print((sum(DEUC3))/(len(DEUC3)))
    print((sum(DEUC4))/(len(DEUC4)))
    print((sum(DEUC5))/(len(DEUC5)))
    print((sum(DEUC6))/(len(DEUC6)))
    print((sum(DEUC7))/(len(DEUC7)))
    print((sum(DEUC8))/(len(DEUC8)))
    print((sum(DEUC9))/(len(DEUC9)))
    print((sum(DEUC10))/(len(DEUC10)))
    print((sum(DEUC11))/(len(DEUC11)))
    print((sum(DEUC12))/(len(DEUC12)))
    print((sum(DEUC13))/(len(DEUC13)))
    print((sum(DEUC14))/(len(DEUC14)))
    print((sum(DEUC15))/(len(DEUC15)))
    print((sum(DEUC16))/(len(DEUC16)))
    print((sum(DEUC17))/(len(DEUC17)))
    print((sum(DEUC18))/(len(DEUC18)))
    print((sum(DEUC19))/(len(DEUC19)))

    plt.boxplot([DEUC0,DEUC1,DEUC2,DEUC3,DEUC4,DEUC5,DEUC6,DEUC7,DEUC8,DEUC9,DEUC10,DEUC11,DEUC12,DEUC13,DEUC14,DEUC15,DEUC16,DEUC17,DEUC18,DEUC19])
    plt.xlabel('particulas')
    plt.ylabel('velocidad')                                                                                                 
    plt.show()
    plt.close()

    plt.scatter(1, p.m[0], c='red')
    plt.scatter(2, p.m[1], c='red')
    plt.scatter(3, p.m[2], c='red')
    plt.scatter(4, p.m[3], c='red')
    plt.scatter(5, p.m[4], c='red')
    plt.scatter(6, p.m[5], c='red')
    plt.scatter(7, p.m[6], c='red')
    plt.scatter(8, p.m[7], c='red')
    plt.scatter(9, p.m[8], c='red')
    plt.scatter(10, p.m[9], c='red')
    plt.scatter(11, p.m[10], c='red')
    plt.scatter(12, p.m[11], c='red')
    plt.scatter(13, p.m[12], c='red')
    plt.scatter(14, p.m[13], c='red')
    plt.scatter(15, p.m[14], c='red')
    plt.scatter(16, p.m[15], c='red')
    plt.scatter(17, p.m[16], c='red')
    plt.scatter(18, p.m[17], c='red')
    plt.scatter(19, p.m[18], c='red')
    plt.scatter(20, p.m[19], c='red')
    plt.xlabel('particulas')
    plt.ylabel('masa')                                                                                              
    plt.show()
    plt.close()

    
    plt.scatter(1, p.c[0], c='green')
    plt.scatter(2, p.c[1], c='green')
    plt.scatter(3, p.c[2], c='green')
    plt.scatter(4, p.c[3], c='green')
    plt.scatter(5, p.c[4], c='green')
    plt.scatter(6, p.c[5], c='green')
    plt.scatter(7, p.c[6], c='green')
    plt.scatter(8, p.c[7], c='green')
    plt.scatter(9, p.c[8], c='green')
    plt.scatter(10, p.c[9], c='green')
    plt.scatter(11, p.c[10], c='green')
    plt.scatter(12, p.c[11], c='green')
    plt.scatter(13, p.c[12], c='green')
    plt.scatter(14, p.c[13], c='green')
    plt.scatter(15, p.c[14], c='green')
    plt.scatter(16, p.c[15], c='green')
    plt.scatter(17, p.c[16], c='green')
    plt.scatter(18, p.c[17], c='green')
    plt.scatter(19, p.c[18], c='green')
    plt.scatter(20, p.c[19], c='green')
    plt.xlabel('particulas')
    plt.ylabel('carga')                                                                                                
    plt.show()
    plt.close()
    popen('convert -delay 50 -size 300x300 p9p_t*.png -loop 0 p9p.gif') # requiere ImageMagick

    













