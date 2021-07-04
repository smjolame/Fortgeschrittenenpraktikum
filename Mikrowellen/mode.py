import numpy as np
import matplotlib.pyplot as plt

def parabel(U,a,b,c):
    return a*U**2+b*U+c
U1_lin = np.linspace(200, 246)
U2_lin = np.linspace(115, 161)
U3_lin = np.linspace(62, 109)
# 1. Mode
V10 = 220
V11 = 210
V12 = 236
A1 = 6.4
V1 = np.array([V10,V11,V12])
Y1 = np.array([A1,A1/2,A1/2])
U1 = np.array([[V10**2,V10,1],[V11**2,V11,1],[V12**2,V12,1]]) 
B1 = np.array([[A1],[A1/2],[A1/2]])
param1 = np.linalg.solve(U1,B1)


# 2.Mode
V20 = 140
V21 = 125
V22 = 151
A2 = 7.35
V2 = np.array([V20,V21,V22])
Y2 = np.array([A2,A2/2,A2/2])
U2 = np.array([[V20**2,V20,1],[V21**2,V21,1],[V22**2,V22,1]]) 
B2 = np.array([[A2],[A2/2],[A2/2]])
param2 = np.linalg.solve(U2,B2)


# 3.Mode 
V30 = 85
V31 = 72
V32 = 99
A3 = 6.2
V3 = np.array([V30,V31,V32])
Y3 = np.array([A3,A3/2,A3/2])
U3 = np.array([[V30**2,V30,1],[V31**2,V31,1],[V32**2,V32,1]]) 
B3 = np.array([[A3],[A3/2],[A3/2]])
param3 = np.linalg.solve(U3,B3)

plt.plot(V1,Y1,'x',label= 'Messwerte 1.Modus', c='k')
plt.plot(U1_lin,parabel(U1_lin,*param1),label= '1. Modus')
plt.plot(V2,Y2,'*',label= 'Messwerte 2.Modus',c='k')
plt.plot(U2_lin,parabel(U2_lin,*param2),label= '2. Modus')
plt.plot(V3,Y3,'o',label= 'Messwerte 3.Modus',c='k')
plt.plot(U3_lin,parabel(U3_lin,*param3),label= '3. Modus')
plt.legend(loc = 'best')
plt.ylim((0,11))
plt.xlabel(r'$U \mathbin{/} \si{\volt}$')
plt.ylabel(r'$y$')
plt.grid()
plt.savefig('build/mode.pdf')
plt.clf()


print('Parameter Mode1: \n',param1)
print('Parameter Mode2: \n',param2)
print('Parameter Mode3: \n',param3)