import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat


L_lin = np.linspace(-30,30,100)

# TEM 00
L_00 , I_00 = np.genfromtxt('data/TEM_00.txt', delimiter=',', unpack=True)


def I_00_fit(L,a,I_0,w):
    return I_0*np.exp(-(L-a)**2/w**2)



#TEM 01
L_01 , I_01 = np.genfromtxt('data/TEM_01.txt', delimiter=',', unpack=True)

def I_01_fit(L,a,I_0,w):
    return 8*(L-a)**2*I_0/(w**2)*np.exp(-(L-a)**2/w**2)



#Curvefit

params_00, cov_00 = curve_fit(I_00_fit, L_00, I_00,p0=[4,0.01,1])
a_0 = ufloat(params_00[0],np.absolute(cov_00[0][0])**0.5)
I_0 = ufloat(params_00[1],np.absolute(cov_00[1][1])**0.5)
w_0 = ufloat(params_00[2],np.absolute(cov_00[2][2])**0.5)
print(f'Für I_00:\n a:{a_0},\n I:{I_0},\n w:{w_0}')

params_01, cov_01 = curve_fit(I_01_fit, L_01, I_01,p0=[8,1,16])
a_1 = ufloat(params_01[0],np.absolute(cov_01[0][0])**0.5)
I_1 = ufloat(params_01[1],np.absolute(cov_01[1][1])**0.5)
w_1 = ufloat(params_01[2],np.absolute(cov_00[2][2])**0.5)
print(f'Für I_01:\n a:{a_1},\n I:{I_1},\n w:{w_1}')


plt.plot(L_00, I_00,'.', label='Messwerte')
plt.plot(L_lin,I_00_fit(L_lin,a_0.n,I_0.n,w_0.n),label='Fitkurve',alpha=0.7)
plt.legend()
plt.grid()
plt.xlabel(r'$l \mathbin{/} \si{\milli\m}$')
plt.ylabel(r'$I \mathbin{/} \si{\micro\watt}$')
plt.savefig('build/TEM_00.pdf')
plt.clf()




plt.plot(L_01, I_01,'.', label='Messwerte')
plt.plot(L_lin,I_01_fit(L_lin,a_1.n,I_1.n,w_1.n),label='Fitkurve',alpha=0.7)
plt.legend()
plt.grid()
plt.xlabel(r'$l \mathbin{/} \si{\milli\m}$')
plt.ylabel(r'$I \mathbin{/} \si{\micro\watt}$')
plt.savefig('build/TEM_01.pdf')
plt.clf()
