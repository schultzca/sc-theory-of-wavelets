import numpy.matlib as M

def haar_matrix(dim):
    '''
    Generate haar transformation matrix.
    '''
    Q = M.matrix('1 1;-1 1')
    return 1 / M.sqrt(2) * M.kron(M.eye(dim / 2), Q)


def permutation_matrix(dim):
    '''
    Generate haar permuation matrix.
    '''
    I = M.eye(dim, dim)
    return M.concatenate((I[::2][:],
                          I[1::2][:]))


def haar_forward(image, dim=None, recursive=True):
    '''
    Apply haar transformation to image.

    :param image: square matrix
    :param dim: dimension
    :param recursive: apply recursively
    :return: None
    '''
    if dim is None:
        dim = M.shape(image)[0]

    if dim == 1:  # base case
        return

    P = permutation_matrix(dim)
    T = haar_matrix(dim)

    image[0:dim, 0:dim] = P * T * M.asmatrix(image[0:dim, 0:dim]) * T.T * P.T

    if recursive:
        haar_forward(image, dim / 2)


def haar_inverse(image, dim=2):
    '''
    Apply haar inverse transformation.

    :param image: square matrix
    :param dim: dimension
    :return: None
    '''
    P = permutation_matrix(dim)
    T = haar_matrix(dim)

    image[0:dim, 0:dim] = T.T * P.T * M.asmatrix(image[0:dim, 0:dim]) * P * T

    if dim == M.shape(image)[0]:
        return

    haar_inverse(image, dim * 2)


