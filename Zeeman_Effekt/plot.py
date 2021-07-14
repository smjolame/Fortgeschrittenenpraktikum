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

def fitfunc(I,a,b,c,d):
    return a*I**3+b*I**2+c*I+d

I_lin = np.linspace(0,5.5)

I , B = np.genfromtxt('data/b_feld.txt', delimiter=',', unpack=True)
params, cov = curve_fit(fitfunc, I, B)
perr = np.sqrt(np.diag(cov))

plt.plot(I, B,'x', label='Messwerte')
plt.plot(I_lin, fitfunc(I_lin,*params), label='Kurve aus Ausgleichsrechnung')
plt.legend()
plt.xlabel(r'$I \mathbin{/} \si{\ampere}$')
plt.ylabel(r'$B \mathbin{/} \si{\milli\tesla}$')
plt.grid()
plt.savefig('build/b_kurve.pdf')

params = uarray(params, perr)
print('verwendete Ausgleichsfunktion: B(I)=aI^3+bI^2+cI+d\n mit den Parametern (a,b,c,d):\n', params)
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
