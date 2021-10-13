import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat

# constants
k_B = 1.380649 * 10**(-23) #J/K
T_0 = 222.4 #K

# function
def tau(T, tau_0, W):
    return tau_0 * np.exp(W/(k_B*T))

# values
W_approx = 1.43e-19  #J
tau_0_approx = 1.4e-13 #s

W_int = 9.4e-20 #J
tau_0_int = 1.8e-12 #s

T_lin = np.linspace(T_0, 280, 1000)

# plot
#fig1 = plt.figure()
#plt.plot(T_lin, tau(T_lin, tau_0_approx, W_approx), label='Methode 1: Näherung, 2.0 K/min')
#plt.xlabel(r'$T \,/\, K$')
#plt.ylabel(r'$\tau (T)$')
#plt.legend()
#plt.tight_layout()
#fig1.savefig('build/relaxationszeit_approx_20.pdf')
#
#fig2 = plt.figure()
#plt.plot(T_lin, tau(T_lin, tau_0_int, W_int), label='Methode 2: Integration, 2.0 K/min')
#plt.xlabel(r'$T \,/\, K$')
#plt.ylabel(r'$\tau (T)$')
#plt.legend()
#plt.tight_layout()
#fig2.savefig('build/relaxationszeit_int_20.pdf')
#