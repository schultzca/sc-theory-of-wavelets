import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import numpy.matlib as M
import utilities.wavelet as wavelet
import utilities.quantization as quantization
import utilities.image as I
import scipy.cluster.vq as vq
from copy import deepcopy
from math import sqrt

h = [1/sqrt(2), 1/sqrt(2)]


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

    fig = plt.gcf()
    fig.suptitle("Problem 1: Haar Transform")

    # Display original matrix
    plt.subplot(321)
    plt.title('original')
    plt.imshow(A, cm.Greys_r)

    plt.subplot(322)
    plt.title('forward')
    wavelet.forward(h, A, recursive=False)
    plt.imshow(A, cm.Greys_r)

    plt.subplot(323)
    plt.title('top left')
    plt.imshow(A[:dim / 2, :dim / 2], cm.Greys_r)

    plt.subplot(324)
    plt.title('top right')
    plt.imshow(A[:dim / 2, dim / 2:], cm.Greys_r)

    plt.subplot(325)
    plt.title('bottom left')
    plt.imshow(A[dim / 2:, :dim / 2], cm.Greys_r)

    plt.subplot(326)
    plt.title('bottom right')
    plt.imshow(A[dim / 2:, dim / 2:], cm.Greys_r)

    plt.show()


def homework_3_problem_3():
    # load image
    img = I.read_gecko_image()
    org = deepcopy(img)

    fig = plt.gcf()
    fig.suptitle("Problem 3: Lloyd's Algorithm")

    plt.subplot(221)
    plt.title('original')
    plt.imshow(org, cm.Greys_r)

    # Apply haar transformation
    wavelet.forward(h, img)
    plt.subplot(222)
    plt.title('forward')
    plt.imshow(img, cm.Greys_r)

    # store sign and apply thresholding
    img, sign = quantization.thresh(img, cutoff=0.8)

    # logarithmic codebook guess
    x = img[np.where(img > 0)]
    init = np.array(
        [min(x) * 2**i for i in range(int(np.ceil(np.log2(max(x)/min(x)))))])

    # Apply k-means to find optimal codebook
    codebook, distortion = vq.kmeans(np.sort(img[np.where(img > 0)]), init)

    # Encode image
    img, dist = vq.vq(img, codebook)

    # Decode image
    img = np.array([s*codebook[i] for s, i in zip(sign, img)])
    img = img.reshape([256, 256])

    plt.subplot(223)
    plt.title('after quant')
    plt.imshow(img, cm.Greys_r)

    # Revert haar transformation
    wavelet.inverse(h, img)

    plt.subplot(224)
    plt.title('reverse')
    plt.imshow(img, cm.Greys_r)
    plt.show()


if __name__ == "__main__":
    homework_3_problem_1()
    homework_3_problem_3()
