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
from scipy import integrate



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

I_dep = I[25:31] - f(T[25:31], a_unt, b_unt, c_unt)*10**(-12) #ampere
print(I_dep)
T_dep = T[25:31]

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

I3_dep = I3[10:16] - f(T3[10:16], a3_unt, b3_unt, c3_unt)*10**(-12) #ampere
print(I3_dep)
T3_dep = T3[10:16]

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
    I_int[i] = integrate.trapz(I_dep[i:], x = T_dep[i:])
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
plt.plot(x, g(x, *params), label = 'Ausgleichsgerade')
plt.ylabel(r'$\ln \, \frac{\int \mathrm{I} \mathrm{dT}}{\mathrm{I}}$')
plt.xlabel(r'$\dfrac{1}{\mathrm{T}} \,/\, K$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('build/integration_15.pdf')

######################################################################################################################################################
##I_dep = I[10:30] - f(T[10:30], a_unt, b_unt, c_unt)*10**(-12) #ampere
##T_dep= T[10:30]
#
## Integration
#I3_int = np.zeros(len(I3_dep))
#for i, I3 in enumerate(I3_dep):
#    I3_int[i] = integrate.trapz(I3_dep[i:], x = T3_dep[i:])
#I3_int[-1] = I3_dep[-1]
#print('I_int =', I3_int)
#
## Ausgleichsgerade
#def g(x, a, b):
#    return a*x + b
#
#y3 = np.log(I3_int/I3_dep)
#x3 = 1/T3_dep
#
##params, cov = curve_fit(g, x, y)
##errors = np.sqrt(np.diag(cov))
##a_ger = ufloat(params[0], errors[0])
##b_ger = ufloat(params[1], errors[1])
##W = a_ger * k_B
#
#print(f'a = {a_ger}')
#print(f'b = {b_ger}')
#print(f'W = a*k_B = {W} J = {W/e_volt} eV')
#
## Plot
#plt.plot(x3, y3, 'mx', label='Messwerte')
#plt.plot(x3, g(x3, *params), label = 'Ausgleichsgerade')
#plt.ylabel(r'$\ln \, \frac{\int \mathrm{I} \mathrm{dT}}{\mathrm{I}}$')
#plt.xlabel(r'$\dfrac{1}{\mathrm{T}} \,/\, K$')
#plt.grid()
#plt.legend()
#plt.tight_layout()
#plt.savefig('build/integration_15.pdf')
#
#
##I4_dep = I[10:30] - f(T[10:30], a3_unt, b3_unt, c3_unt)*10**(-12) #ampere
#T4_dep= T[10:30]
#
## Integration
#I4_int = np.zeros(len(I4_dep))
#for i, I in enumerate(I4_dep):
#    I4_int[i] = integrate.trapz(I4_dep[i:], x4 = T4_dep[i:])
#I4_int[-1] = I4_dep[-1]
#print('I4_int =', I4_int)
#
## Ausgleichsgerade
#def g4(x4, a, b):
#    return a*x + b
#
#y4 = np.log(I4_int/I4_dep)
#x4 = 1/T4_dep
#
#params, cov = curve_fit(g4, x4, y4)
#errors = np.sqrt(np.diag(cov))
#a4_ger = ufloat(params[0], errors[0])
#b4_ger = ufloat(params[1], errors[1])
#W4 = a4_ger * k_B
#
#print(f'a = {a4_ger}')
#print(f'b = {b4_ger}')
#print(f'W4 = a*k_B = {W4} J = {W/e_volt} eV')
#
## Plot
#plt.plot(x4, y4, 'mx', label='Messwerte')
#plt.plot(x4, g4(x4, *params), label = 'Ausgleichsgerade')
#plt.ylabel(r'$\ln \, \frac{\int \mathrm{I} \mathrm{dT}}{\mathrm{I}}$')
#plt.xlabel(r'$\dfrac{1}{\mathrm{T}} \,/\, K$')
#plt.grid()
#plt.legend()
#plt.tight_layout()
#plt.savefig('build/integration_20.pdf')