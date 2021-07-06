import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties.unumpy import uarray
from uncertainties import unumpy as unp
from scipy.stats import sem
from scipy.optimize import curve_fit

lamb = 2 * (108.9 - 84.5) #mm

d1 = 74 #mm
d2 = 72.5 #mm

S_3_exact = np.sqrt(1+1/np.sin(np.pi*(d1 - d2)/lamb)**2)
S_3 = lamb / (np.pi*(d1-d2))

print('SWR aus 3. Exakt:' ,S_3_exact,'SWR aus 3. ungefähr:', S_3)


A1 = 20 #dB
A2 = 43.5 #dB

S_4 = 10**((A2-A1)/20)
print('SWR aus 4. :', S_4)