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
