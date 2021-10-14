import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.unumpy import uarray
from uncertainties import unumpy as unp
from uncertainties.unumpy import (nominal_values as noms,std_devs as stds)

def abw(exact,approx):
    return (exact-approx)*100/exact  #Abweichnung

W_lit = 0.66 #eV

W_1_15 = ufloat(0.91,0.04)
W_1_2 =  ufloat(0.709,0.023)
W_2_15 = ufloat(0.674,0.030)
W_2_2 =  ufloat(0.893,0.023)


print(abw(W_lit,W_1_15))
print(abw(W_lit,W_1_2))
print(abw(W_lit,W_2_15))
print(abw(W_lit,W_2_2))


t_lit = 4 * 10**(-14)
t_1_15 = ufloat(2.5,29) * 10**(-12)
t_1_2 =  ufloat(1.8,21) * 10**(-12)
t_2_15 = ufloat(1.9,32) * 10**(-13)
t_2_2=   ufloat(1.4,24) * 10**(-13)

print(abw(t_lit, t_1_15))
print(abw(t_lit, t_1_2))
print(abw(t_lit, t_2_15))
print(abw(t_lit, t_2_2))
