# pyplot
import matplotlib.pyplot as plt

# numpy imports
import numpy as np
import numpy.matlib as M

# image load, store, display
from matplotlib.image import imread
from matplotlib.cm import Greys_r
from scipy.misc import imresize

# lloyds
import scipy.cluster.vq as vq

# utilities
from copy import deepcopy

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


def _thresh(mat, cutoff=0.8):
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
    thresh, lmaxt = _thresh(img, 0.95)

    # Apply Logarithmic Quantization
    img = encode(img, thresh, lmaxt)

    # Reverse Logarithmic Quantization
    img = decode(img, thresh, lmaxt)

    haar_inverse(img)
    plt.imshow(img)
    plt.show()

def thresh(mat, cutoff=0.8):

    # flatten and sort matrix
    x = np.sort(mat.flatten())

    # find idx corresponding to 80% threshold
    idx = int(np.ceil(len(x)*cutoff))
    
    return np.array([v if v >= x[idx] else 0 for v in mat.flatten()]).reshape(mat.shape)


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
    img = img.astype(float)
    org = deepcopy(img)
    plt.subplot(221)
    plt.title('original')
    plt.imshow(org, Greys_r)

    # Apply haar transformation
    haar_forward(img)
    plt.subplot(222)
    plt.title('forward')
    plt.imshow(img, Greys_r)
    
    # store sign and apply thresholding
    img = img.flatten()    
    sign = [np.sign(v) for v in img]
    img = thresh(abs(img), cutoff=0.8)
  
    # logarithmic codebook guess
    x = img[np.where(img > 0)]
    init = [min(x) * 2**i for i in range(int(np.ceil(np.log2(max(x)/min(x)))))]
    
    # Apply k-means to find optimal codebook
    codebook, distortion = vq.kmeans(np.sort(img[np.where(img > 0)]), init)
    
    # Encode image 
    img, dist = vq.vq(img, codebook) 

    # Decode image
    img = np.array([s*codebook[i] for s, i in zip(sign, img)])
    img = img.reshape([256,256])
    
    plt.subplot(223)
    plt.title('after quant')
    plt.imshow(img, Greys_r)

    # Revert haar transformation
    haar_inverse(img)
    
    plt.subplot(224)
    plt.title('reverse')
    plt.imshow(img, Greys_r)
    plt.show()


if __name__ == "__main__":
    homework_3_problem_3()

