import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib as M
from matplotlib.image import imread
from matplotlib.cm import Greys_r
from scipy.misc import imresize
from copy import deepcopy
import random


def rgb2gray(rgb):
    '''
    Convert rgb image to grayscale
    '''
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


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


def compute_threshold(mat, cutoff=0.8):
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


def log_quant(mat, thresh, lmaxt):
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


def log_quant_inv(mat, thresh, lmaxt):
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


def cluster_points(X, mu):
    clusters = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x - mu[i[0]])) \
                         for i in enumerate(mu)], key=lambda t: t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters


def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis=0))
    return newmu


def has_converged(mu, oldmu):
    return set([a for a in mu]) == set([a for a in oldmu])


def find_centers(X, K):
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return (mu, clusters)


def homework_2():
    # load image
    img = imread('gecko.jpg')
    img = rgb2gray(img)
    img = imresize(img, [256, 256], 'bicubic')
    img = M.asmatrix(img, dtype='float64')  # found out main source of issues!
    org = deepcopy(img)
    plt.imshow(img)
    plt.show()

    haar_forward(img)
    plt.imshow(img)
    plt.show()

    # Compute Threshold
    thresh, lmaxt = compute_threshold(img, 0.95)

    # Apply Logarithmic Quantization
    img = log_quant(img, thresh, lmaxt)

    # Reverse Logarithmic Quantization
    img = log_quant_inv(img, thresh, lmaxt)

    haar_inverse(img)
    plt.imshow(img)
    plt.show()


def homework_3_problem_1():
    # Generate X matrix with border
    dim = 256
    A = M.identity(dim)
    A = A + M.fliplr(A)
    A[0, :] = 1
    A[-1, :] = 1
    A[:, 0] = 1
    A[:, -1] = 1
    A[dim / 2, dim / 2] = 1
    A = A * 255

    # Display original matrix
    plt.title('Original Image')
    plt.imshow(A, Greys_r)
    plt.show()

    plt.title('Haar Transformation')
    haar_forward(A, recursive=False)
    plt.imshow(A, Greys_r)
    plt.show()

    plt.title('Top Left Quadrant')
    plt.imshow(A[:dim / 2, :dim / 2], Greys_r)
    plt.show()

    plt.title('Top Right Quadrant')
    plt.imshow(A[:dim / 2, dim / 2:], Greys_r)
    plt.show()

    plt.title('Bottom Left Quadrant')
    plt.imshow(A[dim / 2:, :dim / 2], Greys_r)
    plt.show()

    plt.title('Bottom Right Quadrant')
    plt.imshow(A[dim / 2:, dim / 2:], Greys_r)
    plt.show()


def homework_3_problem_3():
    dim = 256
    # load image
    img = imread('gecko.jpg')
    img = rgb2gray(img)
    img = imresize(img, [dim, dim], 'bicubic')
    img = M.asmatrix(img, dtype='float64')
    org = deepcopy(img)
    plt.imshow(img, Greys_r)
    plt.show()

    # Apply haar transformation
    haar_forward(img)

    # Convert matrix to 1 dimensional array
    X = np.array([(img[x, y]) for x in range(0, dim) for y in range(0, dim)], dtype='float64')

    # Find centroids and clusters
    [mu, clusters] = find_centers(X, 7)

    # convert clusters to partitions and map to corresponding centroid
    quant = {(min(clusters[i]), max(clusters[i])): int(mu[i]) for i in range(len(mu))}

    # Apply quantization to image.
    for i in range(dim):
        for j in range(dim):
            for key in quant:
                v = img[i, j]
                if (v >= key[0]) and (v <= key[1]):
                    img[i, j] = quant[key]

    # Revert haar transformation
    haar_inverse(img)

    plt.imshow(img, Greys_r)
    plt.show()


if __name__ == "__main__":
    homework_3_problem_1()
    homework_3_problem_3()
