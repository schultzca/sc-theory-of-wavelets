import numpy as np


def log_thresh(mat, cutoff=0.8):
    '''
    Compute threshold and scale used for logarithmic quantization.

    :param mat: matrix input
    :param cutoff: cutoff percent e.g. 085 -> 85%
    :return: threshold, threshold ratio
    '''
    dr, dc = mat.shape
    dim = dr * dc

    x = np.sort(np.fabs(mat).flatten())
    t = x[np.ceil(dim * cutoff)]
    lmaxt = np.log2(max(x)/t)

    return t, lmaxt


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
    mat_quant = np.zeros([dr, dc])
    for i in range(0, dr):
        for j in range(0, dc):
            if np.fabs(mat[i, j]) > thresh:
                sign = mat[i, j] / np.fabs(mat[i, j])
                ln = np.log2(np.fabs(mat[i, j]) / thresh)
                q = np.ceil(127 * ln / lmaxt)
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
    mat_app = np.zeros([dr, dc])
    for i in range(0, dr):
        for j in range(0, dc):
            if np.fabs(mat[i, j]) > 0:
                sign = mat[i, j] / np.fabs(mat[i, j])
                lq = np.fabs(mat[i, j]) * lmaxt / 127
                mat_app[i, j] = sign * thresh * 2 ** lq
    return mat_app


def thresh(image, cutoff=0.8):
    # store sign
    sign = np.array([v/np.abs(v)
                     for v in image.flatten()]).reshape(image.shape)
    # apply thresholding
    x = np.sort(image.flatten())
    idx = int(np.ceil(len(x)*cutoff))
    img_th = np.array([v if v > x[idx] else 0
                       for v in image.flatten()]).reshape(image.shape)
    return img_th, sign
