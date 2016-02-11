import numpy as np
import numpy.matlib as M

def log_thresh(mat, cutoff=0.8):
    '''
    Compute threshold and scale used for logarithmic quantization.

    :param mat: matrix input
    :param cutoff: cutoff percent e.g. 085 -> 85%
    :return: threshold, threshold ratio
    '''
    dr, dc = mat.shape
    dim = dr * dc

    X = M.sort(M.fabs(M.asarray(mat).reshape(dim, )))
    thresh = X[M.ceil(dim * cutoff)]
    lmaxt = M.log2(X[-1] / thresh)

    return thresh, lmaxt


def encode(mat, thresh, lmaxt):
    '''
    Apply logarithmic quantization to matrix.

    floating point -> integer

    :param mat: matrix input
    :param thresh: threshold value
    :param lmaxt: threshold ratio
    :return: quantized matrix
    '''
    dr, dc = mat.shape
    mat_quant = M.zeros([dr, dc])
    for i in range(0, dr):
        for j in range(0, dc):
            if M.fabs(mat[i, j]) > thresh:
                sign = mat[i, j] / M.fabs(mat[i, j])
                ln = M.log2(M.fabs(mat[i, j]) / thresh)
                q = M.ceil(127 * ln / lmaxt)
                mat_quant[i, j] = sign * q
    return mat_quant


def decode(mat, thresh, lmaxt):
    '''
    Reverse logarithmic quantization.

    integer -> floating point

    :param mat: input matrix
    :param thresh: threshold value
    :param lmaxt: threshold ratio
    :return: converted matrix
    '''
    dr, dc = mat.shape
    mat_app = M.zeros([dr, dc])
    for i in range(0, dr):
        for j in range(0, dc):
            if M.fabs(mat[i, j]) > 0:
                sign = mat[i, j] / M.fabs(mat[i, j])
                lq = M.fabs(mat[i, j]) * lmaxt / 127
                mat_app[i, j] = sign * thresh * 2 ** lq
    return mat_app

def thresh(mat, cutoff=0.8):

    # flatten and sort matrix
    x = np.sort(mat.flatten())

    # find idx corresponding to 80% threshold
    idx = int(np.ceil(len(x)*cutoff))
    
    return np.array([v if v >= x[idx] else 0 for v in mat.flatten()]).reshape(mat.shape)
