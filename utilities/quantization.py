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
    tmp = mat.flatten()
    for i in range(len(tmp)):
        s = np.sign(tmp[i])
        v = np.fabs(tmp[i])
        if v > thresh:
            ln = np.log2(v/thresh)
            q = np.ceil(127*ln/lmaxt)
            tmp[i] = s * q
        else:
            tmp[i] = 0
    return tmp.reshape(mat.shape)


def decode(mat, thresh, lmaxt):
    '''
    Reverse logarithmic quantization.

    integer -> floating point

    :param mat: input matrix
    :param thresh: threshold value
    :param lmaxt: threshold ratio
    :return: converted matrix
    '''
    tmp = mat.flatten()
    for i in range(len(tmp)):
        s = np.sign(tmp[i])
        v = np.fabs(tmp[i])
        if v > 0:
            lq = v * lmaxt/127
            tmp[i] = s*thresh*2**lq
    return tmp.reshape(mat.shape)


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
