import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
from uncertainties.unumpy import uarray
from uncertainties import unumpy as unp


def abw(exact,approx):
    return (exact-approx)*100/exact  #Abweichnung

g_b_sig = ufloat(1.83,0.08)
g_b_pi =  ufloat(0.58,0.05)
g_r_sig = ufloat(0.99,0.07) 

g_b_sig_lit = 1.75
g_b_pi_lit = 0.5
g_r_sig_lit = 1

print(abw(g_b_sig_lit,g_b_sig))
print(abw(g_b_pi_lit,g_b_pi))
print(abw(g_r_sig_lit,g_r_sig))
