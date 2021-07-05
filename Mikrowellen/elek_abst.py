import numpy as np
import matplotlib.pyplot as plt

V_a = 85 #Volt
V_b = 75
V_c = 90



f_a = 9018 # Mhz
f_b = 8977
f_c = 9055

breite = f_c - f_b
empf = breite / (V_c - V_b)

print('Breite:', breite,'MHz','Empfindl:', empf, 'MHz/V') 