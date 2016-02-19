from sympy import simplify, symbols, exp, I, pi, sqrt
from sympy.functions.special.delta_functions import Heaviside

def compute_symbol(h):
    """
    Returns wavelet symbol.

    Args: 
        h (iter): filter coefficients

    Returns:
        symbol function (symbolic): P(f)
    """
    f = symbols('f')
    return 1/sqrt(2) * sum([h[i]*exp(-I*2*pi*i*f) for i in range(len(h))]), f

def compute_scaling_function(h, f=None, iters=20):
    """
    Finds scaling function using cascade algorithm.

    Args:
        h (iter): filter coefficients
        f: scaling function guess
        iters (int): number of iterations to perform
    
    Returns:
        scaling function (symbolic): phi(t)
    """
    t = symbols('t')
    if f is None:
        f = Heaviside(t) - Heaviside(t-1)
    
    for iteration in range(iters):
        f = sum([h[i]*sqrt(2)*f.subs(t,2*t-i).evalf() for i in range(len(h))])
   
    return f, t

def compute_mother_wavelet(h, phi, t):
    """
    Returns mother wavelet function.

    Args:
        h (iter): filter coefficients
        phi (func): scaling function
        t (symbol): scaling function symbol

    Returns:
        (psi (func), t): mother wavelet function
    """
    return sum([(-1)**k * h[1-k] *sqrt(2) * phi.subs(t, 2*t - k) for k in range(len(h))]), t
