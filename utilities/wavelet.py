import numpy as np
from sympy import solve, symbols, exp, I, pi, sqrt
from sympy.utilities.lambdify import lambdify


def permutation_matrix(dim):
    '''
    Generate haar permuation matrix.

    Args:
        dim (int): Dimension of square matrix.

    Returns:
        P (matrix): Permutation matrix of dimension (dim x dim)
    '''
    I = np.eye(dim, dim)
    return np.concatenate((I[::2][:],
                           I[1::2][:]))


def wrap_index(idx, dim):
    """
    Helper function that wraps index when greater than maximum dimension.

    Args:
        idx (int): Unwrapped index
        dim (int): Maximum dimension

    Returns:
        idx (int): idx if idx < dim or idx - dim
    """
    if idx < dim:
        return idx
    else:
        return idx - dim


def transform(h, dim):
    """
    Generates wavelet transform matrix of specified dimension.

    Args:
        h (iter): filter coefficients.
        dim (int): dimension of square matrix.

    Returns:
        mat (matrix): Return dim x dim transform matrix.
    """
    mat = np.zeros([dim, dim])
    shift = 0
    for row in range(dim):
        even = row % 2 == 0
        if even and row > 0:
            shift += 2 if shift + 2 <= dim else shift
        coeff = h if even else reversed(h)
        for i, hv in enumerate(coeff):
            idx = wrap_index(i + shift, dim)
            if even:
                mat[row, idx] = hv
            else:
                mat[row, idx] = hv if i % 2 == 0 else -hv
    return mat


def forward(h, image, dim=None, base=1, recursive=True):
    '''
    Apply haar transformation to image. This modifies the input image.

    Args:
        h (iter): filter coefficients
        image (matrix): square matrix input.
        dim (int): dimension of image (optional).
        base (int): base dimension which serves as recursion limit.
        recursive (bool): apply transformation recursively.

    Returns:
        None
    '''
    if dim is None:
        dim = np.shape(image)[0]

    if dim == base:  # base case
        return

    P = permutation_matrix(dim)
    T = transform(h, dim)

    image[0:dim, 0:dim] = P.dot(T).dot(image[0:dim, 0:dim]).dot(T.T).dot(P.T)

    if recursive:
        forward(h, image, dim / 2)


def inverse(h, image, dim=2, recursive=True):
    '''
    Apply haar inverse transformation. This modifies input image.

    Args:
        image (matrix): square matrix input.
        dim (int): starting dimension to apply inverse transformation.
        recursive (bool): apply transformation recursively.

    Returns:
        None
    '''
    P = permutation_matrix(dim)
    T = transform(h, dim)

    image[0:dim, 0:dim] = T.T.dot(P.T).dot(image[0:dim, 0:dim]).dot(P).dot(T)

    if dim == np.shape(image)[0]:
        return

    if recursive:
        inverse(h, image, dim * 2)


def compute_filter_coefficients(n):
    """
    Compute filter coefficients symbolically.

    Args:
        n (int): number of filter coefficients.

    Returns:
        sol (tuple): tuple of lists of solutions (symbolic)
    """
    h_str = ' '.join(['h%s' % v for v in range(n)])
    h = symbols(h_str)

    # Filter Equations
    eq = list()
    eq.append(sum(v**2 for v in h) - 1)
    eq.append(sum(h) - sqrt(2))

    i = 0
    while len(eq) < n:
        eq.append(sum([(-1)**k*k**i*h[k] for k in range(n)]))
        i += 1
    return solve(eq, h)


def compute_symbol(h):
    """
    Returns wavelet symbol.

    Args:
        h (iter): filter coefficients

    Returns:
        symbol function (symbolic): P(f)
    """
    f = symbols('f')
    P = 1/sqrt(2) * sum([h[i]*exp(-I*2*pi*i*f) for i in range(len(h))])
    return lambdify(f, P, 'numpy')
