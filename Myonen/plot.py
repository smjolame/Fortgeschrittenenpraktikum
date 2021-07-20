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

def abw(exact,approx):
    return (exact-approx)*100/exact  #Abweichnung

#verz ist die relative Verzögerung der PMP
#I die Anzahl Impule pro Sekunde
verz , I = np.genfromtxt('data/verz.txt', delimiter=',', unpack=True)

def plat(x,h):
    return 0*x+h

I_err = np.sqrt(I)
I_uarray = uarray(I, I_err)

#Bereich des Plateaus
anf = 11
end = 22
verz_lin = np.linspace(verz[anf],verz[end])

params1, cov1 = curve_fit(plat, verz[anf:end], I[anf:end],sigma=I_err[anf:end],p0=[2])

plt.figure()
plt.errorbar(verz,I, yerr=I_err, fmt='_',capsize=3, label ='Messwerte')
plt.plot(verz_lin, plat(verz_lin,*params1), ls = '--', label = 'Plateau')
plt.grid()
plt.xlabel('T_vz ns')
plt.ylabel('Impulsrate')
plt.legend()
plt.show()

# T ist T_vz in 0.1 micro sekunden
# K ist die Nummer des Kanals
T , K = np.genfromtxt('data/Tvz_kanal.txt', delimiter=',', unpack=True)

T = np.array(T)
T = T * 0.1
print(T,K)
K_lin = np.linspace(0, 100, 100)

def gerade(x,a,b):
   return a*x+b 
params2, cov2 = curve_fit(gerade, K, T,p0=[1,1])

perr2 = np.sqrt(np.diag(cov2))
params_err2 = uarray(params2, perr2)
plt.plot(K,T, 'kx', label = 'Messwerte')
plt.plot(K, gerade(K,*params2), label = 'Ausgleichsgerade')
plt.grid()
plt.xlabel('Kanal')
plt.ylabel('t in micro sek')
plt.legend()
plt.show()
##Curvefit
#def BeispielFunktion(x,a,b):
#    return a*x+b 
#params, cov = curve_fit(BeispielFunktion, x-Werte, y-Werte,sigma=fehler_der_y_werte,p0=[schätzwert_1,#schätzwert_2])
#a = ufloat(params[0],np.absolute(cov[0][0])**0.5)
#b = ufloat(params[1],np.absolute(cov[1][1])**0.5)
#
#
##Json
#Ergebnisse = json.load(open('data/Ergebnisse.json','r'))
#if not 'Name' in Ergebnisse:
#    Ergebnisse['Name'] = {}
#Ergebnisse['Name']['Name des Wertes']=Wert
#
#json.dump(Ergebnisse,open('data/Ergebnisse.json','w'),indent=4)
