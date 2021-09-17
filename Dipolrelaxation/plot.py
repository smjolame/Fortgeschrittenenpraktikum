import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import json
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.unumpy import uarray
from uncertainties import unumpy as unp
from uncertainties.unumpy import (nominal_values as noms,std_devs as stds)
from scipy.stats import sem
from scipy.integrate import trapz
from scipy import constants
from uncertainties.unumpy import exp



def abw(exact,approx):
    return (exact-approx)*100/exact  #Abweichnung

t1, T1, I1 = np.genfromtxt("data/relax_strom_1.txt", delimiter=",", unpack=True)
t2, T2, I2 = np.genfromtxt("data/relax_strom_2.txt", delimiter=",", unpack=True)
I1 = -I1*10
I2 = -I2*10
T1 = constants.convert_temperature(T1, 'Celsius', 'Kelvin')
T2 = constants.convert_temperature(T2, 'Celsius', 'Kelvin')


#Bestimmung der Heizraten
h1 = np.diff(T1)
h2 = np.diff(T2)

print("h1 mw" , np.mean(h1))
print("h1 f" , sem(h1))
print("h2 mw" , np.mean(h2))
print("h2 f" , sem(h2))

print('######################')
##########################


#Bestimmung der Untergrundrate
def f(x, a, b, c):
    return a * np.exp(b*x)+ c

I1_unter = np.append(I1[0:25], I1[45:71])
T1_unter = np.append(T1[0:25], T1[45:71])
print('I1_untergrund:', I1_unter)
print('T1_untergrund:', T1_unter)
params, cov = curve_fit(f, T1_unter, I1_unter, p0=[10**(-5), 0.01, 0.2])
errors = np.sqrt(np.diag(cov))

a1 = ufloat(params[0], errors[0])
b1 = ufloat(params[1], errors[1])
c1 = ufloat(params[2], errors[2])

print('a1 =', a1)
print('b1 =', b1)
print('c1 =', c1)
print('########################')

T1_lin = np.linspace(T1_unter.min(), T1_unter.max(), 1000)

# depolarisationsstrom
I1_pol = I1[25:45]
T1_pol = T1[25:45]

#plot
plt.plot(T1, I1, 'mx', label='Messwerte')
plt.plot(T1_unter, I1_unter, 'mo', alpha=0.2, label='Teilmenge: Untergrund')
plt.plot(T1_pol, I1_pol, 'go', alpha=0.4, label='Teilmenge: Depolarisationsstrom')
plt.plot(T1_lin, f(T1_lin, *params), label='Ausgleichskurve: Untergrund')

plt.xlabel(r'$T \,/\, K$')
plt.ylabel(r'$I \,/\, pA$')
plt.legend()
plt.tight_layout()

plt.savefig('build/T1_test.pdf')
plt.clf()

#######################################################################################################################################################################################
I2_unter = np.append(I2[0:10], I2[28:50])
T2_unter = np.append(T2[0:10], T2[28:50])
print('I2_untergrund:', I2_unter)
print('T2_untergrund:', T2_unter)

params, cov = curve_fit(f, T2_unter, I2_unter , p0=[10**(-5), 0.01, 0.2])
errors = np.sqrt(np.diag(cov))

a2 = ufloat(params[0], errors[0])
b2 = ufloat(params[1], errors[1])
c2 = ufloat(params[2], errors[2])

print('a2 =', a2)
print('b2 =', b2)
print('c2 =', c2)
print('########################')

T2_lin = np.linspace(T2_unter.min(), T2_unter.max(), 1000)

# depolarisationsstrom
I2_pol = I2[10:28]
T2_pol = T2[10:28]

#plot
plt.plot(T2, I2, 'mx', label='Messwerte')
plt.plot(T2_unter, I2_unter, 'mo', alpha=0.3, label='Untergrund')
plt.plot(T2_pol, I2_pol, 'go', alpha=0.6, label='Depolarisationsstrom')
plt.plot(T2_lin, f(T2_lin, *params), label='Ausgleichsfunktion Untergrund')

plt.xlabel(r'$T \,/\, K$')
plt.ylabel(r'$I \,/\, pA$')
plt.legend()
plt.tight_layout()

plt.savefig('build/T2_test.pdf')
plt.clf()

#######################################################################################################################################################################################

t, T, I = np.genfromtxt('data/relax_strom_1.txt', comments='#', unpack=True, delimiter=',')
I = -I*10**(-11) #pA
T = constants.convert_temperature(T, 'Celsius', 'Kelvin')
# d = 3 * 10**(-3) # m
# A = np.pi * (d/2)**2
# I= I / A 

k_B = 1.380649 * 10**(-23) #J/K
e_volt = 1.602176634 * 10**(-19) #J
a_unt = 1.2*10**(-6)
b_unt = 0.0543
c_unt = 0.75

def f(x, a, b, c):
    return a * np.exp(b*x)+ c

def I_func(T, a, W):
    return a * np.exp(-W / (k_B *T))

I_dep = I[25:35] - f(T[25:35], a_unt, b_unt, c_unt)*10**(-12) #ampere
print(I_dep)
T_dep = T[25:35]

params, cov = curve_fit(I_func, T_dep, I_dep, p0 = [1, 10**(-19)] )#p0=[1, 10**(-19)])
errors = np.sqrt(np.diag(cov))
W = ufloat(params[1], errors[1])
print(f'Aktivierungsenergie W approx 15 = {W}J = {W/e_volt}eV')

T_lin = np.linspace(T_dep.min(), T_dep.max(), 10000)
plt.plot(T_dep, I_dep, 'mx', label='Messwerte')
plt.plot(T_lin, I_func(T_lin, *params), label='Regressionsfunktion')
plt.xlabel(r'$T \,/\, K$')
plt.ylabel(r'$I \,/\, pA$')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('build/W_approx_15.pdf')
plt.clf()
#plt.show()

###################################################################################################################################################
t3, T3, I3 = np.genfromtxt('data/relax_strom_2.txt', comments='#', unpack=True, delimiter=',')
I3 = -I3*10**(-11) #pA
T3 = constants.convert_temperature(T3, 'Celsius', 'Kelvin')
# d = 3 * 10**(-3) # m
# A = np.pi * (d/2)**2
# I= I / A 

k_B = 1.380649 * 10**(-23) #J/K
e_volt = 1.602176634 * 10**(-19) #J
a3_unt = 2.7*10**(-6)
b3_unt = 0.0525
c3_unt = 0.8

def f(x, a, b, c):
    return a * np.exp(b*x)+ c

def I3_func(T, a, W):
    return a * np.exp(-W / (k_B *T))

I3_dep = I3[10:20] - f(T3[10:20], a3_unt, b3_unt, c3_unt)*10**(-12) #ampere
print(I3_dep)
T3_dep = T3[10:20]

params3, cov3 = curve_fit(I3_func, T3_dep, I3_dep, p0 = [1, 10**(-19)] )#p0=[1, 10**(-19)])
errors3 = np.sqrt(np.diag(cov))
W3 = ufloat(params3[1], errors3[1])
print(f'Aktivierungsenergie W approx 20 = {W3}J = {W3/e_volt}eV')

T3_lin = np.linspace(T3_dep.min(), T3_dep.max(), 10000)
plt.plot(T3_dep, I3_dep, 'mx', label='Messwerte')
plt.plot(T3_lin, I3_func(T3_lin, *params3), label='Regressionsfunktion')
plt.xlabel(r'$T \,/\, K$')
plt.ylabel(r'$I \,/\, pA$')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('build/W_approx_20.pdf')
plt.clf()
#
####################################################################################################################################################

I_dep = I[25:45] - f(T[25:45], a_unt, b_unt, c_unt)*10**(-12) #ampere
T_dep= T[25:45]

# Integration
I_int = np.zeros(len(I_dep))
for i, I in enumerate(I_dep):
    I_int[i] = trapz(I_dep[i:], x = T_dep[i:])
I_int[-1] = I_dep[-1]
print('I_int =', I_int)

# Ausgleichsgerade
def g(x, a, b):
    return a*x + b

y = np.log(I_int/I_dep)
x = 1/T_dep

params, cov = curve_fit(g, x, y)
errors = np.sqrt(np.diag(cov))
a_ger = ufloat(params[0], errors[0])
b_ger = ufloat(params[1], errors[1])
W = a_ger * k_B

print(f'a = {a_ger}')
print(f'b = {b_ger}')
print(f'W = a*k_B = {W} J = {W/e_volt} eV')

# Plot
plt.plot(x, y, 'mx', label='Messwerte')
plt.plot(x, g(x, *params), label = 'Lineare Regression')
plt.ylabel(r'$\ln \, \frac{\int \mathrm{I} \mathrm{dT}}{\mathrm{I}}$')
plt.xlabel(r'$\dfrac{1}{\mathrm{T}} \,/\, K$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('build/integration_15.pdf')

#####################################################################################################################################################
#
#I3_dep = I[10:30] - f(T[10:30], a3_unt, b3_unt, c3_unt)*10**(-12) #ampere
#T3_dep= T[10:30]
#
## Integration
#I3_int = np.zeros(len(I3_dep))
#for i, I in enumerate(I3_dep):
#    I3_int[i] = trapz(I3_dep[i:], x = T3_dep[i:])
#I3_int[-1] = I3_dep[-1]
#print('I_int =', I3_int)
#
## Ausgleichsgerade
#def g(x, a, b):
#    return a*x + b
#
#y = np.log(I3_int/I3_dep)
#x = 1/T3_dep
#
#params, cov = curve_fit(g, x, y)
#errors = np.sqrt(np.diag(cov))
#a_ger = ufloat(params[0], errors[0])
#b_ger = ufloat(params[1], errors[1])
#W = a_ger * k_B
#
#print(f'a = {a_ger}')
#print(f'b = {b_ger}')
#print(f'W = a*k_B = {W} J = {W/e_volt} eV')
#
## Plot
#plt.plot(x, y, 'mx', label='Messwerte')
#plt.plot(x, g(x, *params), label = 'Ausgleichsgerade')
#plt.ylabel(r'$\ln \, \frac{\int \mathrm{I(T)} \mathrm{dT}}{\mathrm{I}}$')
#plt.xlabel(r'$\dfrac{1}{\mathrm{T}} \,/\, K$')
#plt.grid()
#plt.legend()
#plt.tight_layout()
#plt.savefig('build/integration_20.pdf')
##################################################################################################################################################


#t = t*60
#params, cov = curve_fit(f, t, T)
#errors = np.sqrt(np.diag(cov))
#b = ufloat(params[0], errors[0]) #heizrate
#T_0 = ufloat(params[1], errors[1])
#
#print(f'b = {b} K/s = {b*60} K/min')
#print(f'T_0 = {T_0}')
#
## Berechnung char. relaxationszeit
#k_B = 1.380649 * 10**(-23) #J/K
#T_max = 260 #K
##W_int = ufloat(1.17*10**(-19),0.04*10**(-19))
#W_approx = ufloat(1.26*10**(-19), 0.06*10**(-19))
#
#tau_max_approx = k_B * T_max**2 / (b*W_approx)
##tau_max_int = k_B * T_max**2 / (b*W_int)
##print(f'tau_int = {tau_max_int} s')
#print(f'tau_approx = {tau_max_approx} s')
#
#tau_0_approx = tau_max_approx * unp.exp(-W_approx / (k_B*T_max))
##tau_0_int = tau_max_int * unp.exp(-W_int / (k_B*T_max))
#
#print(f'tau_0 für approx: {tau_0_approx}')
##print(f'tau_0 für int: {tau_0_int}')
#
## Plot
#plt.plot(t/60, T, 'mx', label='Messwerte')
#plt.plot(t/60, f(t, *params), label='Ausgleichsgerade')
#plt.ylabel(r'$T \,/\, K$')
#plt.xlabel(r'$t \,/\, min$')
#plt.grid()
#plt.legend()
#plt.tight_layout()
#plt.savefig('build/char_relaxationszeit_15.pdf')
##plt.show()