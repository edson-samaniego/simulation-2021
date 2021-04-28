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
        dire = 1
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
    popen('convert -delay 50 -size 300x300 p9p_t*.png -loop 0 p9p.gif') # requiere ImageMagick
