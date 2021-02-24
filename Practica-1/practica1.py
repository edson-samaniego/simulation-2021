from random import random, randint
import matplotlib.pyplot as plt
from tabulate import tabulate
import cv2 as cv2

porcentaje = []
promedio = []
minimo = []
maximo = []

for A in range(1, 6):
    dim = A
    
    for pot in range(4, 10):
        saltos = 2**pot
        rep= 30
        resultados = []
        nuevos = []# porque sino lista se hace con none y no puedo extraer numeros
        datos=[],[],[],[],[],[]
        pos = [0] * dim
        
        for replica in range(rep):
            nunca = True
            for paso in range(saltos):
                cual = randint(0, dim -1)
                pos[cual] = pos[cual] + 1 if random()<0.5 else pos[cual] -1
                if all([p== 0 for p in pos]):
                    resultados.append(paso)
                    nuevos.append(paso)
                    nunca = False
                    break
            if nunca:
                resultados.append(None)

        cuantos = sum([r is None for r in resultados])
        porc = ((cuantos/rep)*100)#obtencion de porcentaje
        porcentaje.append(porc) #acumulo los porcentajes por potencia
        #print(porc,'% no regreso',A,pot)
       
        if cuantos< rep:
            regresaron = sum([r if r is not None else 0 for r in resultados])
            prom = (regresaron / (rep - cuantos))#obtencion de promedio
            promedio.append(prom)                #acumulo promedios por potencia
            #print(prom,'promedio',A,pot)
            for i in nuevos:
                datos[A].append(i)
            #print(datos[A])
            minimo.append(min(datos[A]))
            maximo.append(max(datos[A]))
                                     
        else:
            #print('no hay')
            promedio.append(0)#estos append es para que acumule un cero donde no regreso
            minimo.append(0)
            maximo.append(0)
        #print(minimo)
        #print(maximo)
      #cv2.waitKey(10000) solo se utilizo para pausar ciclos y analizar

      
####potencia 4
Tabla1 = [["dimension", "minimo", "maximo", "promedio", "porcentaje"],
        ["1", minimo[0], maximo[0], promedio[0], porcentaje[0]],
        ["2", minimo[6], maximo[6], promedio[6], porcentaje[6]],
        ["3", minimo[12], maximo[12], promedio[12], porcentaje[12]],
        ["4", minimo[18], maximo[18], promedio[18], porcentaje[18]],
        ["5", minimo[24], maximo[24], promedio[24], porcentaje[24]]]

#### potencia 5
Tabla2 = [["dimension", "minimo", "maximo", "promedio", "porcentaje"],
        ["1", minimo[1], maximo[1], promedio[1], porcentaje[1]],
        ["2", minimo[7], maximo[7], promedio[7], porcentaje[7]],
        ["3", minimo[13], maximo[13], promedio[13], porcentaje[13]],
        ["4", minimo[19], maximo[19], promedio[19], porcentaje[19]],
        ["5", minimo[25], maximo[25], promedio[25], porcentaje[25]]]

#### potencia 6
Tabla3 = [["dimension", "minimo", "maximo", "promedio", "porcentaje"],
        ["1", minimo[2], maximo[2], promedio[2], porcentaje[2]],
        ["2", minimo[8], maximo[8], promedio[8], porcentaje[8]],
        ["3", minimo[14], maximo[14], promedio[14], porcentaje[14]],
        ["4", minimo[20], maximo[20], promedio[20], porcentaje[20]],
        ["5", minimo[26], maximo[26], promedio[26], porcentaje[26]]]

#### potencia 7
Tabla4 = [["dimension", "minimo", "maximo", "promedio", "porcentaje"],
        ["1", minimo[3], maximo[3], promedio[3], porcentaje[3]],
        ["2", minimo[9], maximo[9], promedio[9], porcentaje[9]],
        ["3", minimo[15], maximo[15], promedio[15], porcentaje[15]],
        ["4", minimo[21], maximo[21], promedio[21], porcentaje[21]],
        ["5", minimo[27], maximo[27], promedio[27], porcentaje[27]]]

#### potencia 8
Tabla5 = [["dimension", "minimo", "maximo", "promedio", "porcentaje"],
        ["1", minimo[4], maximo[4], promedio[4], porcentaje[4]],
        ["2", minimo[10], maximo[10], promedio[10], porcentaje[10]],
        ["3", minimo[16], maximo[16], promedio[16], porcentaje[16]],
        ["4", minimo[22], maximo[22], promedio[22], porcentaje[22]],
        ["5", minimo[28], maximo[28], promedio[28], porcentaje[28]]]

#### potencia 9
Tabla6 = [["dimension", "minimo", "maximo", "promedio", "porcentaje"],
        ["1", minimo[5], maximo[5], promedio[5], porcentaje[5]],
        ["2", minimo[11], maximo[11], promedio[11], porcentaje[11]],
        ["3", minimo[17], maximo[17], promedio[17], porcentaje[17]],
        ["4", minimo[23], maximo[23], promedio[23], porcentaje[23]],
        ["5", minimo[29], maximo[29], promedio[29], porcentaje[29]]]

print('pasos 16')
print(tabulate(Tabla1))
print('pasos 32')
print(tabulate(Tabla2))
print('pasos 64')
print(tabulate(Tabla3))
print('pasos 128')
print(tabulate(Tabla4))
print('pasos 256')
print(tabulate(Tabla5))
print('pasos 512')
print(tabulate(Tabla6))

d1 = [(maximo[0]),(minimo[0]),(promedio[0])]
d2 = [(maximo[6]),(minimo[6]),(promedio[6])]
d3 = [(maximo[12]),(minimo[12]),(promedio[12])]
d4 = [(maximo[18]),(minimo[18]),(promedio[18])]
d5 = [(maximo[24]),(minimo[24]),(promedio[24])]
plt.boxplot([d1, d2, d3, d4, d5])
plt.savefig('16_pasos.png')
plt.show()
d6 = [(maximo[1]),(minimo[1]),(promedio[1])]
d7 = [(maximo[7]),(minimo[7]),(promedio[7])]
d8 = [(maximo[13]),(minimo[13]),(promedio[13])]
d9 = [(maximo[19]),(minimo[19]),(promedio[19])]
d10 = [(maximo[25]),(minimo[25]),(promedio[25])]
plt.boxplot([d6, d7, d8, d9, d10])
plt.savefig('32_pasos.png')
plt.show()
d11 = [(maximo[2]),(minimo[2]),(promedio[2])]
d12 = [(maximo[8]),(minimo[8]),(promedio[8])]
d13 = [(maximo[14]),(minimo[14]),(promedio[14])]
d14 = [(maximo[20]),(minimo[20]),(promedio[20])]
d15 = [(maximo[26]),(minimo[26]),(promedio[26])]
plt.boxplot([d11, d12, d13, d14, d15])
plt.savefig('64_pasos.png')
plt.show()
d16 = [(maximo[3]),(minimo[3]),(promedio[3])]
d17 = [(maximo[9]),(minimo[9]),(promedio[9])]
d18 = [(maximo[15]),(minimo[15]),(promedio[15])]
d19 = [(maximo[21]),(minimo[21]),(promedio[21])]
d20 = [(maximo[27]),(minimo[27]),(promedio[27])]
plt.boxplot([d16, d17, d18, d19, d20])
plt.savefig('128_pasos.png')
plt.show()
d21 = [(maximo[4]),(minimo[4]),(promedio[4])]
d22 = [(maximo[10]),(minimo[10]),(promedio[10])]
d23 = [(maximo[16]),(minimo[16]),(promedio[16])]
d24 = [(maximo[22]),(minimo[22]),(promedio[22])]
d25 = [(maximo[28]),(minimo[28]),(promedio[28])]
plt.boxplot([d21, d22, d23, d24, d25])
plt.savefig('256_pasos.png')
plt.show()
d26 = [(maximo[5]),(minimo[5]),(promedio[5])]
d27 = [(maximo[11]),(minimo[11]),(promedio[11])]
d28 = [(maximo[17]),(minimo[17]),(promedio[17])]
d29 = [(maximo[23]),(minimo[23]),(promedio[23])]
d30 = [(maximo[29]),(minimo[29]),(promedio[29])]
plt.boxplot([d26, d27, d28, d29, d30])
plt.savefig('512_pasos.png')
plt.show()




