import numpy as np
import matplotlib.pyplot as plt

abst_kon , I_kon = np.genfromtxt('data/kon_kon.txt',delimiter=',', unpack=True)
abst_plan , I_plan = np.genfromtxt('data/plan_kon.txt', delimiter=',',unpack=True)
linspce = np.linspace(0,6,100)


plt. plot(abst_kon, I_kon,'o', label='Messwerte')
plt.xlabel(r'$L \mathbin{/}\si{\centi\m}$')
plt.ylabel(r'$I \mathbin{/}\si{\milli\watt}$')
#plt.axvline(x=140, ls='--',c='k', alpha=0.9,label='theo. Stabilitätsgrenze')
plt.legend()
plt.grid()
plt.savefig('build/kon.pdf')
plt.clf()

plt. plot(abst_plan, I_plan,'o', label='Messwerte')
plt.xlabel(r'$L \mathbin{/}\si{\centi\m}$')
plt.ylabel(r'$I \mathbin{/}\si{\milli\watt}$')
plt.axvline(x=140, ls='--',c='k', alpha=0.9,label='theo. Stabilitätsgrenze')
plt.legend()
plt.grid()
plt.savefig('build/plan.pdf')
plt.clf()