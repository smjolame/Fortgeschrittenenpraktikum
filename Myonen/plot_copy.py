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

#verz ist die relative Verzögerung der PMP (ns)
#I die Anzahl Impule pro Sekunde
verz , I = np.genfromtxt('data/verz.txt', delimiter=',', unpack=True)

def plat(x,h):
    return 0*x+h

I_err = np.sqrt(I)
I_uarray = uarray(I, I_err)

#Bereich des Plateaus
anf = 11
end = 22
verz_lin1 = np.linspace(verz[anf],verz[end])

params1, cov1 = curve_fit(plat, verz[anf:end], I[anf:end],sigma=I_err[anf:end],p0=[2])

h_halb = params1/2


halb_1 = -7
halb_2 = 7

verz_halb1 = verz[7]
verz_halb2 = verz[25]
verz_lin2 = np.linspace(verz_halb1,verz_halb2)

halbwertbreite = np.abs(verz_halb1-verz_halb2)
print('Halbwertsbreite:', halbwertbreite)


print('Höhe Plateau:', params1)
print('Ränder der Halbwertsbreite:' ,verz[7], verz[25])


# T ist T_vz in 0.1 micro sekunden
# K ist die Nummer des Kanals
T , K = np.genfromtxt('data/Tvz_kanal.txt', delimiter=',', unpack=True)

T = np.array(T)
T = T * 0.1

K_lin = np.linspace(0, 100, 100)

def gerade(x,a,b):
   return a*x+b 
params2, cov2 = curve_fit(gerade, K, T,p0=[1,1])

perr2 = np.sqrt(np.diag(cov2))
params2_err2 = uarray(params2, perr2)
print('Parameter der Ausgleichsgerade:', params2_err2)



#Lebensdauer Kram
N_start = 3256768
N_stopp = 17775
T_mess = 175726 #s
kanal = np.genfromtxt('data/kanaele.txt', unpack=True)

T_such = 20 *10**(-6) #s
anzahl_kanaele = len(kanal)
anzahl_kanaele_gef = len(kanal[kanal>0])
print('Kanäle:',anzahl_kanaele)
print('Kanäle gefüllt:',anzahl_kanaele_gef)
nu = N_start/T_mess
print('nu:',nu)
P_1 = T_such * nu * np.exp(T_such*nu)
print('P(1):',P_1)
N_Untergrund = N_start * P_1
print('Anzahl Fehlmessungen:', N_Untergrund)
I_Untergrund = N_Untergrund/anzahl_kanaele_gef 
print('Untergrundrate pro Kanal:', I_Untergrund)

t = gerade(range(len(kanal)), *params2)
t_lin = np.linspace(0,17)
def N(t,N_0,lamb,I):
    return N_0*np.exp(-lamb*t)+I

kanal_bereinigt = kanal[5:]

kanal_bereinigt = np.append(kanal[1], kanal_bereinigt)

cut = 150
params3, cov3 = curve_fit(N, t[:cut] , kanal_bereinigt[:cut] ,p0=[400,0.5,10])
params3_err3 = uarray(params3, np.sqrt(np.diag(cov3)))
print('Parameter der Fitfunktion (N_0, lamb, I):', params3_err3)

plt.errorbar(t[:cut], kanal[:cut],c = 'k', yerr = np.sqrt(kanal[:cut]),fmt='_',capsize=3, label ='Messwerte', ecolor = 'y' )
plt.plot(t_lin, N(t_lin, *params3), label = 'Ausgleichskurve', c = 'b')
plt.yscale('log')
plt.xlabel(r'$t \:/\: \mu s $' )
plt.ylabel(r'$N$')
plt.legend()
plt.grid()
#plt.xlim(0,20)
plt.savefig('build/lebensdauer.pdf')
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
