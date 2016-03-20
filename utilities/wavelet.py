import numpy as np
import sympy as sy
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
    n = 6
    h = sy.symbols(' '.join(['h%s' % i for i in range(n)]))
    eq = list()
    eq.append(sum(h)-sy.sqrt(2))
    eq.append(sum([v**2 for v in h]) - 1)

    if n > 2:
        for k in range(1, n):
            eq.append(sum([h[i]*h[i+2*k]
                           for i in range(n) if i + 2*k <= n - 1]))
        for k in range(n/2):
            eq.append(sum([(-1)**m*(m**k)*h[m] for m in range(n)]))

    solutions = sy.solve(eq, h)

    return solutions


def compute_symbol(h):
    """
    Returns wavelet symbol.

    Args:
        h (iter): filter coefficients

    Returns:
        symbol function (symbolic): P(f)
    """
    f = sy.symbols('f')
    P = 1/sy.sqrt(2) * sum([h[i]*sy.exp(-sy.I*2*sy.pi*i*f)
                            for i in range(len(h))])
    return lambdify(f, P, 'numpy')


def battle_lemarie_symbol():
    """
    Symbolic B-L symbol
    """
    f = sy.symbols('f')

    phi_hat = battle_lemarie_scaling_transform()

    # symbolic expression of symbol
    symbol = phi_hat.subs(f, 2*f)/phi_hat
    symbol.simplify()

    return symbol


def battle_lemarie_scaling_transform():
    """
    Symbolic fourier transform of B-L scaling function
    """
    w, f = sy.symbols('w, f')

    # symbolic expression of the scaling function
    w = 2*sy.pi*f
    N1 = 5 + 30*sy.cos(w/2)**2 + 30*(sy.sin(w/2)**2)*(sy.cos(w/2)**2)
    N2 = 2*sy.sin(w/2)**4 * sy.cos(w/2)**2 + 70 * \
        sy.cos(w/2)**4 + (2/3)*sy.sin(w/2)**6
    S = (N1 + N2)/(105*sy.sin(w/2)**8)
    phi_hat = 16/(w**4*sy.sqrt(S))
    phi_hat.simplify()

    return phi_hat


def battle_lemarie_wavelet_transform():
    """
    Symbolic fourier transform of B-L wavelet
    """
    f = sy.symbols('f')

    symbol = battle_lemarie_symbol()
    phi_hat = battle_lemarie_scaling_transform()

    psi_hat = -sy.exp(-sy.I*sy.pi*f) * \
        sy.conjugate(symbol.subs(f, (1+f)/2))*phi_hat.subs(f, f/2)

    return psi_hat


def cdf_24_encoding_transform(n=256):
    """
    CDF 2.4 Encoding Transform Matrix

    Args:
        n (int): desired dimension

    Returns:
        ndarray [nxn]: CDF 2.4 transform matrix
    """
    # low pass filter coefficients
    L = np.array([0.03314563036812,
                  -0.06629126073624,
                  -0.17677669529664,
                  0.41984465132951,
                  0.99436891104358,
                  0.41984465132951,
                  -0.17677669529664,
                  -0.06629126073624,
                  0.03314563036812])

    # high pass filter coefficients
    H = np.array([0,
                  0,
                  0,
                  0.35355339059327,
                  -0.70710678118655,
                  0.35355339059327,
                  0,
                  0,
                  0])

    T = np.zeros([n, n])

    offset = 4
    row_size = n + 2*offset
    for i in range(n):
        tmp = np.zeros([row_size])
        end = i + L.size
        end = end if end < row_size else row_size
        if i % 2 == 0:
            tmp[i:end] = L[:end-i]
        else:
            tmp[i:end] = H[:end-i]

        T[i, :] = tmp[offset:row_size-offset]
        T[i, 1:offset+1] += np.flipud(tmp[:offset])
        T[i, -5:-1] += np.flipud(tmp[-offset:])

    return T


def cdf_24_decoding_transform(n=256):
    """
    CDF 2.4 Decoding Transform Matrix

    Args:
        n (int): desired dimension

    Returns:
        ndarray [nxn]: CDF 2.4 transform matrix
    """
    # low pass filter coefficients
    H = np.array([-0.03314563036812,
                  -0.06629126073624,
                  0.17677669529664,
                  0.41984465132951,
                  -0.99436891104358,
                  0.41984465132951,
                  0.17677669529664,
                  -0.06629126073624,
                  -0.03314563036812])

    # high pass filter coefficients
    L = np.array([0,
                  0,
                  0,
                  0.35355339059327,
                  0.70710678118655,
                  0.35355339059327,
                  0,
                  0,
                  0])

    offset = 4
    row_size = n + 2*offset
    T = np.zeros([n, n])
    for i in range(n):
        tmp = np.zeros([row_size])
        end = i + L.size
        end = end if end < row_size else row_size
        if i % 2 == 0:
            tmp[i:end] = L[:end-i]
        else:
            tmp[i:end] = H[:end-i]
        T[i, :] = tmp[offset:row_size-offset]

    l = L[-4:]
    h = H[-4:]

    for i in range(4):
        if i % 2 == 0:
            T[i+1, :(4-i)] += h[-(4-i):]
            T[(n-2)-i, -(4-i):] += np.flipud(l[-(4-i):])
        else:
            T[i+1, :(4-i)] += l[-(4-i):]
            T[(n-2)-i, -(4-i):] += np.flipud(h[-(4-i):])
    return T
