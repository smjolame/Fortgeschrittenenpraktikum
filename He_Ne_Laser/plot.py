import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import json
from scipy.optimize import curve_fit
#from uncertainties import ufloat
#from uncertainties.unumpy import uarray
#from uncertainties import unumpy as unp
#from uncertainties.unumpy import (nominal_values as noms,std_devs as stds)
from scipy.stats import sem

def abw(exact,approx):
    return (exact-approx)*100/exact  #Abweichnung

abst_kon , I_kon = np.genfromtxt('data/kon_kon.txt',delimiter=',', unpack=True)
abst_plan, I_plan = np.genfromtxt('data/plan_kon.txt', delimiter=',',unpack=True)

plt. plot(abst_kon, I_kon,'o', label='Konkav Konkav')
plt.xlabel('bla')
plt.ylabel('bla')
plt.legend()
plt.grid()
plt.show()
plt.clf()

plt. plot(abst_plan, I_plan,'o', label='Plan Konkav')
plt.xlabel('bla')
plt.ylabel('bla')
plt.legend()
plt.grid()
plt.show()
plt.clf()

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
