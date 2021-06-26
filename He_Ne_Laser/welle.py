import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.npyio import genfromtxt
from uncertainties import ufloat
from uncertainties.unumpy import uarray
from uncertainties import unumpy as unp
from uncertainties.unumpy import (nominal_values as noms,std_devs as stds)
from scipy.stats import sem
err = 0.05  #cm

g1 = 1/600 *10**(6) #nm
g2 = 1/100 *10**(6) #nm
g3 = 1/1200 *10**(6) #nm
g4 = 1/80 *10**(6) #nm

l1, n1 = genfromtxt('data/Gitter01.txt', delimiter=',', unpack=True) #cm
L1 = ufloat(82,err) #cm
l2, n2 = genfromtxt('data/Gitter02.txt', delimiter=',', unpack=True) #cm
L2 = ufloat(74,err) #cm
l3, n3 = genfromtxt('data/Gitter03.txt', delimiter=',', unpack=True) #cm
L3 = ufloat(29,err) #cm
l4, n4 = genfromtxt('data/Gitter04.txt', delimiter=',', unpack=True) #cm
L4 = ufloat(77,err) #cm

l1_u = uarray(l1,err)
l2_u = uarray(l2,err)
l3_u = uarray(l3,err)
l4_u = uarray(l4,err)

def lamb(l,L,g,n):
    return (g*unp.sin(unp.arctan(l/L)))/n

lamb1_array = lamb(l1_u,L1,g1,n1)
lamb2_array = lamb(l2_u,L2,g2,n2)
lamb3_array = lamb(l3_u,L3,g3,n3)
lamb4_array = lamb(l4_u,L4,g4,n4)

lamb1 = np.mean(lamb1_array)
lamb2 = np.mean(lamb2_array)
lamb3 = np.mean(lamb3_array)
lamb4 = np.mean(lamb4_array)
lamb_fin = np.mean([lamb1, lamb2, lamb3, lamb4])
sem_ = sem([lamb1.n, lamb2.n, lamb3.n, lamb4.n])

lamb_fin_u = ufloat(lamb_fin.n, sem_)
print(lamb_fin_u)