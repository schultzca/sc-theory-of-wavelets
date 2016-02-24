from sympy import symbols, sqrt, solve
from scipy.signal.wavelets import cascade
import matplotlib.pyplot as plt
import numpy as np
import utilities.wavelet as wv


def compute_A3_filter_coefficeints():

    h = symbols('h0 h1 h2 h3 h4 h5')

    # Filter Equations
    eq1 = sum(v**2 for v in h) - 1
    eq2 = sum([h[i-2]*h[i] for i in range(2, len(h))])
    eq3 = sum(h) - sqrt(2)

    # A(3) Equations
    eq4 = sum([(-1)**i*h[i] for i in range(0, len(h))])
    eq5 = sum([(-1)**i*(-i)*h[i] for i in range(0, len(h))])
    eq6 = sum([(-1)**i*i**2*h[i] for i in range(0, len(h))])

    solutions = solve([eq1, eq2, eq3, eq4, eq5, eq6], *h)
    print solutions


def plot_A3_symbol(show=True):
    h = [0.332670552950083, 0.806891509311093, 0.459877502118492, -
         0.135011020010255, -0.0854412738820267, 0.0352262918857095]

    # Compute Symbol
    P, f = wv.compute_symbol(h)

    plt.figure(1)

    # Plot Symbol
    t = np.linspace(0, 1, 200)
    plt.plot(t, [abs(P.subs(f, v).evalf()) for v in t])
    plt.xlabel('f')
    plt.ylabel('abs(P)')
    plt.title('Symbol P(f)')

    if show:
        plt.show()

def plot_A3_scaling_function_and_wavelet(show=True):
    h = [0.332670552950083, 0.806891509311093, 0.459877502118492, -
         0.135011020010255, -0.0854412738820267, 0.0352262918857095]

    plt.figure(2)

    t, phi, psi = cascade(h)
    plt.subplot(1,2,1)
    plt.plot(t, phi)
    plt.xlabel('t')
    plt.ylabel('phi')
    plt.title('scaling function')

    plt.subplot(1,2,2)
    plt.plot(t, psi)
    plt.xlabel('t')
    plt.ylabel('psi')
    plt.title('mother wavelet')

    if show:
        plt.show()


if __name__ == "__main__":
    plot_A3_symbol(show=False)
    plot_A3_scaling_function_and_wavelet()
