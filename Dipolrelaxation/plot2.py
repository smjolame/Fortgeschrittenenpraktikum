import numpy as np
from matplotlib import pyplot as plt
from uncertainties import ufloat
from scipy.optimize import curve_fit
from scipy import integrate

# values
k_B = 1.380649 * 10**(-23) #J/K
e_volt = 1.602176634 * 10**(-19) #J
a_unt = 2.7e-06
b_unt = 0.0525
c_unt = 0.80

t, T, I = np.genfromtxt('data/relax_strom_2.txt', unpack=True, delimiter=',')
#I_offset = 3.5 #pA #+ I_offset # strom vorzeichen (definition) + offsett
I = -I * 10#**(-12) #ampere
T = T + 273.15 # Kelvin

def f(x, a, b, c):
    return a * np.exp(b*x)+ c

# Polarisationsstrom
I_dep = I[10:30] - f(T[10:30], a_unt, b_unt, c_unt)*10**(-12) #ampere
T_dep= T[10:30]

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
plt.savefig('build/integration_20.pdf')
#plt.show()