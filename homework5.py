import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.wavelets import cascade
from sympy import symbols, conjugate, sqrt, solve, evaluate, exp, I, pi
import utilities.wavelet as wv

# declare variables
h1, h2, h3, h4 = symbols("h1 h2 h3 h4")

# initial constraint
eq0 = h1 - 2*h2 + 3*h3  # = 0

# wavelet equation 1
eq1 = h1**2 + h2**2 + h3**2 + h4**2 - 1 # = 0

# wavelet equation 2
eq3 = h1*conjugate(h3) + h2*conjugate(h4)   # = 0

# wavelet equation 3
eq2 = h1 + h2 + h3 + h4 -sqrt(2)  # = 0

# wavelet equation 4
eq4 = -h1 + h2 - h3 + h4    # = 0

solutions = solve([eq0, eq1, eq2, eq3, eq4], [h1, h2, h3, h4])

# verify solutions satisfy equations
var_list = [h1, h2, h3, h4] 
for solution in solutions:
    par = zip(var_list, solution)
    assert eq1.subs(par).evalf(chop=True) == 0 # satisfy eq0
    assert eq1.subs(par).evalf(chop=True) == 0 # satisfy eq1
    assert eq2.subs(par).evalf(chop=True) == 0 # satisfy eq2
    assert eq3.subs(par).evalf(chop=True) == 0 # satisfy eq3
    assert eq4.subs(par).evalf(chop=True) == 0 # satisfy eq4

h = [v.evalf() for v in solutions[0]]

# Compute Symbol
P, f = wv.compute_symbol(h)

plt.figure(1)

# Plot Symbol
t = np.linspace(0,1,100)
plt.plot(t, [abs(P.subs(f, v).evalf()) for v in t])
plt.xlabel('f')
plt.ylabel('abs(P)')
plt.title('Symbol P(f)')

plt.figure(2)

# Compute Scaling Function Symbolically
time = np.linspace(0,3,100)
plt.subplot(1,2,1)
phi, t = wv.compute_scaling_function(h, iters=7)
plt.plot(time, [phi.subs(t,v).evalf() for v in time])
plt.xlabel('t')
plt.ylabel('phi')
plt.title('phi (%s iterations)' % 7)

plt.subplot(1,2,2)
psi, t = wv.compute_mother_wavelet(h, phi, t)
plt.plot(time, [psi.subs(t,v).evalf() for v in time])
plt.xlabel('t')
plt.ylabel('psi')
plt.title('psi (%s iterations)' % 7)

plt.figure(3)

# Compute Scaling Function and Wavelet Numerically
t, phi, psi = cascade(h)
plt.subplot(1,2,1)
plt.plot(t, phi)
plt.xlabel('t')
plt.ylabel('phi')
plt.title('phi (scipy)')

plt.subplot(1,2,2)
plt.plot(t, psi)
plt.xlabel('t')
plt.ylabel('psi')
plt.title('psi (scipy)')
plt.show()

