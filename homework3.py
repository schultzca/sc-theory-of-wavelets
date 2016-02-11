import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib as M

import utilities.haar as H
import utilities.quantization as Q
import utilities.image as I

import scipy.cluster.vq as vq

from copy import deepcopy

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
    plt.imshow(A, I.Greys_r)

    plt.subplot(322)
    plt.title('forward')
    H.haar_forward(A, recursive=False)
    plt.imshow(A, I.Greys_r)

    plt.subplot(323)
    plt.title('top left')
    plt.imshow(A[:dim / 2, :dim / 2], I.Greys_r)

    plt.subplot(324)
    plt.title('top right')
    plt.imshow(A[:dim / 2, dim / 2:], I.Greys_r)

    plt.subplot(325)
    plt.title('bottom left')
    plt.imshow(A[dim / 2:, :dim / 2], I.Greys_r)

    plt.subplot(326)
    plt.title('bottom right')
    plt.imshow(A[dim / 2:, dim / 2:], I.Greys_r)

    plt.show()

def homework_3_problem_3():
    dim = 256
    # load image
    img = I.imread('gecko.jpg')
    img = I.rgb2gray(img)
    img = I.imresize(img, [dim, dim], 'bicubic')
    img = img.astype(float)
    org = deepcopy(img)
    
    fig = plt.gcf()
    fig.suptitle("Problem 3: Lloyd's Algorithm")

    plt.subplot(221)
    plt.title('original')
    plt.imshow(org, I.Greys_r)

    # Apply haar transformation
    H.haar_forward(img)
    plt.subplot(222)
    plt.title('forward')
    plt.imshow(img, I.Greys_r)

    # store sign and apply thresholding
    img = img.flatten()    
    sign = [np.sign(v) for v in img]
    img = Q.thresh(abs(img), cutoff=0.8)

    # logarithmic codebook guess
    x = img[np.where(img > 0)]
    init = np.array([min(x) * 2**i for i in range(int(np.ceil(np.log2(max(x)/min(x)))))])

    # Apply k-means to find optimal codebook
    codebook, distortion = vq.kmeans(np.sort(img[np.where(img > 0)]), init)

    # Encode image 
    img, dist = vq.vq(img, codebook) 

    # Decode image
    img = np.array([s*codebook[i] for s, i in zip(sign, img)])
    img = img.reshape([256,256])

    plt.subplot(223)
    plt.title('after quant')
    plt.imshow(img, I.Greys_r)

    # Revert haar transformation
    H.haar_inverse(img)

    plt.subplot(224)
    plt.title('reverse')
    plt.imshow(img, I.Greys_r)
    plt.show()


if __name__ == "__main__":
    homework_3_problem_1()
    homework_3_problem_3()
