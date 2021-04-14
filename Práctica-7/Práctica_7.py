
import matplotlib.pyplot as plt
from random import uniform
from math import sqrt, fabs
import numpy as np
import math as ma 
from cv2 import cv2

def g(x, y):
    return (((x + 0.5)**4 - 30 * x**2 - 20 * x + (y + 0.5)**4 - 30 * y**2 - 20 * y)**2)/100

low = -6
high = 5.5
step = 0.20
tmax = 30
px=[]
py=[]
mx=[]
my=[]
dato=[]
for ciclo in range(tmax):
    
    currx = uniform(low, high)
    curry = uniform(low, high)
    [bestx, besty] = [currx, curry]
    for iteracion in range(tmax):
        
        deltax = uniform(0, step)
        leftx = currx - deltax  
        leftx = low if leftx < low else leftx  
        rightx = currx + deltax 
        rightx = high if rightx > high else rightx
        
        deltay = uniform(0, step)
        lefty = curry - deltay  
        lefty = low if lefty < low else lefty  
        righty = curry + deltay  
        righty = high if righty > high else righty

        [currx, curry] = [leftx, lefty] if g(leftx, lefty) > g(leftx, righty) else [leftx, righty]
        [currx, curry] = [currx, curry] if g(currx, curry) > g(rightx, righty) else [rightx, righty]  
        [currx, curry] = [currx, curry] if g(currx, curry) > g(rightx, lefty) else [rightx, lefty] 

        if g(currx, curry) > g(bestx, besty):
            [bestx, besty] = [currx, curry]

############################# grafica
        p = np.arange(low-step, high-step, step)
        n = len(p)
        z = np.zeros((n, n), dtype=float)
        for i in range(n):
            x = p[i]
            for j in range(n):
                y = p[n - j - 1]  
                z[i, j] = g(x, y)
                
        px.append(currx // step - low // step)
        py.append(curry // step - low // step)
        mx.append(bestx // step - low // step)
        my.append(besty // step - low // step)
            
for img in range(tmax):
    t = range(0, n, 5)
    l = ['{:.1f}'.format(low + i * step) for i in t]
    fig, ax = plt.subplots(figsize=(6, 5), ncols=1)
    pos = ax.imshow(z)
    plt.xticks(t, l)
    plt.yticks(t, l[::-1])  
############## 
    for a in range(0, 871, (tmax+img)):        
        dato.append([fabs(px[a]-48)+fabs(py[a]-9), (px[a],py[a])])

    coordenada=min(dato)
    menor=coordenada[1] 
    XM=menor[0]
    YM=menor[1]
##############

    ax.scatter(XM, YM, marker='x', color='red', s=85)
    
    ax.scatter(px[0+img], py[0+img], marker='o', color='red', s=8)    
    ax.scatter(px[30+img], py[30+img], marker='o', color='red', s=8)    
    ax.scatter(px[60+img], py[60+img], marker='o', color='red', s=8)    
    ax.scatter(px[90+img], py[90+img], marker='o', color='red', s=8)    
    ax.scatter(px[120+img], py[120+img], marker='o', color='red', s=8)    
    ax.scatter(px[150+img], py[150+img], marker='o', color='red', s=8)    
    ax.scatter(px[180+img], py[180+img], marker='o', color='red', s=8)    
    ax.scatter(px[210+img], py[210+img], marker='o', color='red', s=8)    
    ax.scatter(px[240+img], py[240+img], marker='o', color='red', s=8)    
    ax.scatter(px[270+img], py[270+img], marker='o', color='red', s=8)    
    ax.scatter(px[300+img], py[300+img], marker='o', color='red', s=8)    
    ax.scatter(px[330+img], py[330+img], marker='o', color='red', s=8)    
    ax.scatter(px[360+img], py[360+img], marker='o', color='red', s=8)    
    ax.scatter(px[390+img], py[390+img], marker='o', color='red', s=8)    
    ax.scatter(px[420+img], py[420+img], marker='o', color='red', s=8)
    ax.scatter(px[450+img], py[450+img], marker='o', color='red', s=8)
    ax.scatter(px[480+img], py[480+img], marker='o', color='red', s=8)
    ax.scatter(px[510+img], py[510+img], marker='o', color='red', s=8)
    ax.scatter(px[540+img], py[540+img], marker='o', color='red', s=8)
    ax.scatter(px[570+img], py[570+img], marker='o', color='red', s=8)
    ax.scatter(px[600+img], py[600+img], marker='o', color='red', s=8)
    ax.scatter(px[630+img], py[630+img], marker='o', color='red', s=8)
    ax.scatter(px[660+img], py[660+img], marker='o', color='red', s=8)
    ax.scatter(px[690+img], py[690+img], marker='o', color='red', s=8)
    ax.scatter(px[720+img], py[720+img], marker='o', color='red', s=8)
    ax.scatter(px[750+img], py[750+img], marker='o', color='red', s=8)
    ax.scatter(px[780+img], py[780+img], marker='o', color='red', s=8)
    ax.scatter(px[810+img], py[810+img], marker='o', color='red', s=8)
    ax.scatter(px[840+img], py[840+img], marker='o', color='red', s=8)
    ax.scatter(px[870+img], py[870+img], marker='o', color='red', s=8)

    fig.colorbar(pos, ax=ax)
    plt.title('{:d} paso'.format(img+1))
    fig.savefig('p7p_{:d}.png'.format(img), bbox_inches='tight')
    plt.close()
    #cv2.waitKey(40000)



















