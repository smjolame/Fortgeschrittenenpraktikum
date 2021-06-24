import numpy as np 
import matplotlib.pyplot as plt 

d_lin = np.linspace(0,3000,10000)
g1 = 1

b2 = 1000
g2 = lambda d: 1-d/b2

b3 = 1400
g3 = lambda d: 1-d/b3

plt.plot(d_lin,g1*g2(d_lin),label = (r'$g_1 \cdot g_2$'))
plt.plot(d_lin,g2(d_lin)*g2(d_lin), label = (r'$g_2 \cdot g_2$'))
plt.plot(d_lin,g2(d_lin)*g3(d_lin), label = (r'$g_2 \cdot g_3$'))
plt.plot(d_lin,g3(d_lin)*g3(d_lin), label = (r'$g_3 \cdot g_3$'))
plt.grid()
plt.xlabel(r'$L$ in mm')
plt.legend()
plt.fill_between(d_lin,0,1, alpha = 0.45)

plt.savefig('build/Vorbereitung_Stabilität.pdf')

#plan g1
#konkav g3
