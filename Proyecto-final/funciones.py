from math import sqrt, fabs
import numpy as np
import pandas as pd
from random import uniform, random
import random

def calentamiento(mezcla, p, p2,afecta,afecta2,temp,m,n,masa):
    mezcla=mezcla
    calorx=0
    calory=0
    trans_calor= mezcla[2]
    dist_centro=[]
    for img in range(0,50):
        DE= []
        DE2= []
        cercanos=0
        dopaje=0
        for i in range(0,n):
            fila= p.iloc[i] #cambio con for la lista por fila 
            xp=fila.x
            yp=fila.y
            eucl=sqrt((xp-calorx)**2+(yp-calory)**2)
            DE.append(eucl)
        if (any([d<afecta for d in DE]))== True:
            cercanos=sum([d<afecta for d in DE])
        if mezcla[2] == .1:    
            for j in range(0,m):
                fila2= p2.iloc[j] 
                xp2=fila2.xcer
                yp2=fila2.ycer
                eucl2=sqrt((xp2-calorx)**2+(yp2-calory)**2)
                DE2.append(eucl2)
            if (any([e<afecta2 for e in DE2]))== True:
                dopaje=sum([e<afecta2 for e in DE2])
        if mezcla[2] ==.06:    
            for j in range(0,m):
                fila2= p2.iloc[j] 
                xp2=fila2.xpol
                yp2=fila2.ypol
                eucl2=sqrt((xp2-calorx)**2+(yp2-calory)**2)
                DE2.append(eucl2)
            if (any([e<afecta2 for e in DE2]))== True:
                dopaje=sum([e<afecta2 for e in DE2])
        if mezcla[2] == .03:    
            for j in range(0,m):
                fila2= p2.iloc[j] 
                xp2=fila2.xplata
                yp2=fila2.yplata
                eucl2=sqrt((xp2-calorx)**2+(yp2-calory)**2)
                DE2.append(eucl2)
            if (any([e<afecta2 for e in DE2]))== True:
                dopaje=sum([e<afecta2 for e in DE2])
        if cercanos==0:
            calorx=(calorx+1+(temp*.003))if calorx< 50 else(calorx-1-(temp*.003))
            calory=(calory+1+(temp*.003))if calory< 50 else(calory-1-(temp*.003))   
        if cercanos >0:
            factor= cercanos * masa # este factor influye en la temp
            calorx=((calorx+1+(temp*.003))-factor)if calorx< 50 else((calorx-1-(temp*.003))+factor)
            calory=((calory+1+(temp*.003))-factor)if calory< 50 else((calory-1-(temp*.003))+factor)
            if dopaje > 0:
                factorC= dopaje * trans_calor
                factorM= dopaje * masa
                calorx=(calorx-(factorC+factorM)) if calorx< 50 else (calorx+(factorC+factorM))
                calory=(calory-(factorC+factorM)) if calorx< 50 else (calory+(factorC+factorM))
        mejor=(sqrt((50-calorx)**2+(50-calory)**2))
        dist_centro.append(mejor)
    return (dist_centro)

def flujo_elec(mezcla,p,p2,afecta,afecta2,temp,m,n,masa):
    mezcla=mezcla
    electx=0
    electy=100
    trans_elect= mezcla[2] ## modificar a flujo electrico
    dist_centro=[]
    for img in range(0,50):
        DE= []
        DE2= []
        cercanos=0
        dopaje=0
        for i in range(0,n):
            fila= p.iloc[i] #cambio con for la lista por fila 
            xp=fila.x
            yp=fila.y
            eucl=sqrt((xp-electx)**2+(yp-electy)**2)
            DE.append(eucl)
        if (any([d<afecta for d in DE]))== True:
            cercanos=sum([d<afecta for d in DE])
        if mezcla[2] == .01:    ## cambiar unidad puntual
            for j in range(0,m):
                fila2= p2.iloc[j] 
                xp2=fila2.xcer
                yp2=fila2.ycer
                eucl2=sqrt((xp2-electx)**2+(yp2-electy)**2)
                DE2.append(eucl2)
            if (any([e<afecta2 for e in DE2]))== True:
                dopaje=sum([e<afecta2 for e in DE2])
        if mezcla[2] ==.04:    ## cambiar unidad puntual
            for j in range(0,m):
                fila2= p2.iloc[j] 
                xp2=fila2.xpol
                yp2=fila2.ypol
                eucl2=sqrt((xp2-electx)**2+(yp2-electy)**2)
                DE2.append(eucl2)
            if (any([e<afecta2 for e in DE2]))== True:
                dopaje=sum([e<afecta2 for e in DE2])
        if mezcla[2] == .1:   ## cambiar unidad puntual
            for j in range(0,m):
                fila2= p2.iloc[j] 
                xp2=fila2.xplata
                yp2=fila2.yplata
                eucl2=sqrt((xp2-electx)**2+(yp2-electy)**2)
                DE2.append(eucl2)
            if (any([e<afecta2 for e in DE2]))== True:
                dopaje=sum([e<afecta2 for e in DE2])
                
        if cercanos==0:
            electx=(electx+1+(temp*.005))if electx< 50 else(electx-1-(temp*.005))
            electy=(electy+1+(temp*.005))if electy< 50 else(electy-1-(temp*.005))   
        if cercanos >0:
            factor= cercanos * masa # este factor influye en la temp
            electx=((electx+1+(temp*.005))-factor)if electx< 50 else((electx-1-(temp*.005))+factor)
            electy=((electy+1+(temp*.005))-factor)if electy< 50 else((electy-1-(temp*.005))+factor)
            if dopaje > 0:
                factor2= dopaje * trans_elect
                factorM= dopaje * masa 
                electx=(electx+(factor2+factorM)) if electx< 50 else (electx-(factor2+factorM))
                electy=(electy+(factor2+factorM)) if electx< 50 else (electy-(factor2+factorM))
        mejor=(sqrt((50-electx)**2+(50-electy)**2))
        dist_centro.append(mejor)
    return(dist_centro)
