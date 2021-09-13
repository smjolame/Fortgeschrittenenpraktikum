import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import json
from scipy.optimize import curve_fit
from scipy.stats.stats import PearsonRConstantInputWarning
from uncertainties import ufloat
from uncertainties.unumpy import uarray
from uncertainties import unumpy as unp
from uncertainties.unumpy import (nominal_values as noms,std_devs as stds)
from scipy.stats import sem


#const
c = 299792458 #m/s
h = 6.62607015e-34 #Js
mu_B = 9.2740100783e-24 #J/T

lamb_blau = 480 #nm
lamb_rot = 643.8 #nm

# blau 
#orte der Maxima in anzahl pixel:
b = np.array([939, 1055, 1177, 1306, 1440, 1579, 1725, 1879, 2042, 2217, 2408, 2616])
#Differenz aller Einträge
b_roh = np.diff(b)
b_roh_err = uarray(b_roh,4)
b_sig = np.array([1080-1028, 1206-1147, 1335-1278, 1469-1408, 1611-1547, 1758-1692, 1912-1843, 2080-2004, 2258-2174, 2451-2365 ,2666-2569])
b_sig_err = uarray(b_sig,4)
b_pi = np.array([1063-1038, 1185-1160, 1315-1289, 1447-1420, 1589-1562, 1734-1706, 1892-1857, 2055-2021, 2231-2195, 2424-2384, 2633-2591])
b_pi_err = uarray(b_pi,4)
#print('b_roh_err',b_roh_err,'\n\n','b_sig_err',b_sig_err,'\n\n','b_pi_err',b_pi_err,'\n\n')

Delta_lamb_blau = 26.95 #pm

#rot 
#analog
r = np.array([1079, 1193, 1307, 1424, 1546, 1668, 1796, 1926, 2061, 2201, 2346, 2497])
r_roh = np.diff(r)
r_roh_err = uarray(r_roh,4)
r_sig = np.array([991-951, 1101-1059, 1215-1173, 1329-1288, 1448-1404, 1569-1524, 1692-1648, 1822-1775, 1953-1905, 2089-2040, 2227-2178])
r_sig_err = uarray(r_sig,4)
#print('r_roh_err',r_roh_err,'\n\n','r_sig_err',r_sig_err,'\n\n')


Delta_lamb_rot = 48.9  #pm


#Berechnung von del lamb

def del_lamb(del_s,Delta_s,Delta_lamb):
    return 0.5 * (del_s/Delta_s)*Delta_lamb

del_lamb_blau_sig = del_lamb(b_sig_err, b_roh_err, Delta_lamb_blau) #pm
del_lamb_blau_pi = del_lamb(b_pi_err, b_roh_err, Delta_lamb_blau) #pm 
del_lamb_rot = del_lamb(r_sig_err, r_roh_err, Delta_lamb_rot) #pm
#print('del_lamb_blau_sig',del_lamb_blau_sig,'\n\n','del_lamb_blau_pi',del_lamb_blau_pi,'\n\n','del_lamb_rot',del_lamb_rot,'\n\n')

del_lamb_blau_sig = np.mean(del_lamb_blau_sig) #pm
del_lamb_blau_pi = np.mean(del_lamb_blau_pi) #pm
del_lamb_rot_sig = np.mean(del_lamb_rot) #pm
#print('del_lamb_blau_sig_mean',del_lamb_blau_sig,'\n\n', 'del_lamb_blau_pi_mean',del_lamb_blau_pi,'\n\n', 'del_lamb_rot_sig_mean', del_lamb_rot_sig,'\n\n' )

#lande:

def fitfunc(I,a,b,c,d):
    return a*I**3+b*I**2+c*I+d

I_lin = np.linspace(0,5.5)

I , B = np.genfromtxt('data/b_feld.txt', delimiter=',', unpack=True)
params, cov = curve_fit(fitfunc, I, B)
perr = np.sqrt(np.diag(cov))
params_err = uarray(params,perr)

def g(B,lamb,del_lamb):
    return 1/(mu_B*B)*(h*c)/(lamb**2)*del_lamb

B_blau_sig = fitfunc(3.33,*params_err)*10**(-3) #T
B_blau_pi = fitfunc(5.03,*params_err) *10**(-3) #T
B_rot_sig = fitfunc(5.03,*params_err)*10**(-3) #T
print('B_blau_sig',B_blau_sig,'\n\n', 'B_blau_pi', B_blau_pi,'\n\n', 'B_rot_sig',B_rot_sig,'\n\n')


g_b_sig = g(B_blau_sig,lamb_blau*10**(-9),del_lamb_blau_sig * 10**(-12))
g_b_pi = g(B_blau_pi,lamb_blau*10**(-9),del_lamb_blau_pi * 10**(-12))
g_r_sig = g(B_rot_sig,lamb_rot*10**(-9),del_lamb_rot_sig * 10**(-12))

print('g_b_sig',g_b_sig,'\n\n', 'g_b_pi',g_b_pi,'\n\n', 'g_r_sig', g_r_sig,'\n\n')