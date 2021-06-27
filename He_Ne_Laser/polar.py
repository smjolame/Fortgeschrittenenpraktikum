import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat

phi , I = np.genfromtxt('data/Polar.txt', delimiter=',',unpack=True)
phi = np.radians(phi)
phi_lin = np.linspace(0,np.pi,100)



def I_fit(p, p_0, I_0):
    return I_0 * (np.cos(p-p_0))**2

params, cov = curve_fit(I_fit, phi, I)
p_0 = ufloat(params[0],np.absolute(cov[0][0])**0.5)
I_0 = ufloat(params[1],np.absolute(cov[1][1])**0.5)
print(p_0,I_0)

plt.plot(phi, I,'o', label='Messwerte')
plt.plot(phi_lin, I_fit(phi_lin, p_0.n, I_0.n), label='Fitkurve')
plt.xlabel(r'$\phi \mathbin{/} \text{rad}$')
plt.ylabel(r'$I \mathbin{/} \si{\micro\watt}$')
plt.grid()
plt.legend()
plt.savefig('build/polar.pdf')
