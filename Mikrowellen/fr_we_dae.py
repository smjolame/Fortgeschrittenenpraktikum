import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties.unumpy import uarray
from uncertainties import unumpy as unp
from scipy.stats import sem
from scipy.optimize import curve_fit

einst_lin1 = np.linspace(0,5)
einst_lin2 = np.linspace(0,2.4)

f_mess = 9006 #MHz
lamb_array =np.array([ 2 * (97 - 72.5), 2 * (72.5-48.1)]) # mm
lamb = ufloat(np.mean(lamb_array), sem(lamb_array) )
a = ufloat(22.860,0.046)
c = 3 * 10**11 #mm/s
f_rech = c * unp.sqrt((1/lamb)**2+(1/(2*a))**2) * 10**(-6) #MHz
print('f gemessen:', f_mess,'MHz;' , 'Wellenlänge:', lamb, 'mm;' , 'f berechnet:' , f_rech,'MHz' )

P = np.array([0,2,4,6,8,10]) #dB
einst = np.array([2.69, 2.93, 3.11, 3.27, 3.44, 3.58]) #mm
herst = np.array([0, 0.9, 1.45, 1.75, 2.03, 2.28]) #mm
def parabel(einst,a,b,c):
    return a*einst**2+b*einst+c
params , cov =  curve_fit(parabel, herst, P)
perr = np.sqrt(np.diag(cov))
params_err_p = np.array([params[0]+perr[0],params[1]+perr[1],params[2]+perr[2]])
params_err_m = np.array([params[0]-perr[0],params[1]-perr[1],params[2]-perr[2]])


plt.plot(einst,P,'x', label = 'Messwerte', c='r')
# Einfügen einer Korrektur von 14, damit die Messwerte auf die Parabel passen
plt.plot(einst,P+14,'x', c='k', label = 'Messwerte nach Korrektur')
plt.fill_between(einst_lin1, parabel(einst_lin1, *params_err_m), parabel(einst_lin1, *params_err_p), alpha = 0.2,label = '1-$\sigma$ Umgebung')
plt.plot(herst, P, 'x', label = 'Herstellerangaben', c='b')
plt.plot(einst_lin1, parabel(einst_lin1, *params), label= 'Fit-Parabel')
plt.grid()
plt.xlabel(r'$x \mathbin{/} \si{\milli\m}$')
plt.ylabel(r'$P \mathbin{/} \si{\decibel}$')
plt.legend()
plt.savefig('build/daempf_weit')
plt.clf()

plt.fill_between(einst_lin2, parabel(einst_lin2, *params_err_m), parabel(einst_lin2, *params_err_p), alpha = 0.2,label = '1-$\sigma$ Umgebung')
plt.plot(herst, P, 'x', label = 'Herstellerangaben',c ='b')
plt.plot(einst_lin2, parabel(einst_lin2, *params), label= 'Fit-Parabel')
plt.xlabel(r'$x \mathbin{/} \si{\milli\m}$')
plt.ylabel(r'$P \mathbin{/} \si{\decibel}$')
plt.grid()
plt.legend()
plt.savefig('build/daempf_nah')
