import numpy as np
import pandas as pd
from random import randint, random
import seaborn as sns
from cv2 import cv2


def poli(maxdeg, varcount, termcount):
    f = []
    for t in range(termcount):
        var = randint(0, varcount - 1)
        deg = randint(1, maxdeg)
        f.append({'var': var, 'coef': random(), 'deg': deg})
    return pd.DataFrame(f)
  
def evaluate(pol, var):
    return sum([t.coef * var[pol.at[i, 'var']]**t.deg for i, t in pol.iterrows()])
 
 
def domin_by(target, challenger):
    if np.any(challenger < target):
        return False
    return np.any(challenger > target)
 
vc = 4
md = 3
tc = 5
pc_violin=[]
#k = 2 # cuantas funciones objetivo
for k in range(2, 9, 2):
    n = 500 # cuantas soluciones aleatorias
    replicas=30
    porcentaje=[]
    #pc_violin=[]
    for rep in range(0, replicas):
        obj = [poli(md, vc, tc) for i in range(k)]
        minim = np.random.rand(2) > 0.5
        sol = np.random.rand(n, vc)
        val = np.zeros((n, 2))
        for i in range(n): # evaluamos las soluciones
            for j in range(2):
                val[i, j] = evaluate(obj[j], sol[i])
        sign = [1 + -2 * m for m in minim]
        mejor1 = np.argmax(sign[0] * val[:, 0])
        mejor2 = np.argmax(sign[1] * val[:, 1])
        cual = {True: 'min', False: 'max'}

        dom = []
        for i in range(n):
            d = [domin_by(sign * val[i], sign * val[j]) for j in range(n)]  
            dom.append(sum(d))
        frente = val[[d == 0 for d in dom], :]
        porc=(len(frente)*100)/n
        porcentaje.append(porc)
    #print(porcentaje)
    pc_violin.append(porcentaje)
    #print(pc_violin)
    cv2.waitKey(1000)
    print('termina k')
#print(len(pc_violin))
print('terminó')
############################################# gráficos
import matplotlib.pyplot as plt

df=pd.DataFrame(
    {"Función Objetivo": replicas * ["02"] + replicas * ["04"] + replicas * ["06"] + replicas * ["08"],
     "Porciento Pareto": pc_violin[0] + pc_violin[1] + pc_violin[2] + pc_violin[3]}
     )
print(df)   
pd.set_option("display.max_rows", None, "display.max_columns", None)
sns.violinplot(x='Función Objetivo', y='Porciento Pareto', data=df, scale='count', cut = 0, palette="Set3")
sns.swarmplot(x="Función Objetivo", y="Porciento Pareto", data=df, color="green")
#plt.boxplot(range(2,9,2), pc_violin)
plt.show()



