import matplotlib.pyplot as plt

import utilities.haar as H
import utilities.quantization as Q
import utilities.image as I

import numpy as np
import numpy.matrixlib as M

from copy import deepcopy

# load image
img = I.imread('gecko.jpg')
img = I.rgb2gray(img)
img = I.imresize(img, [256, 256], 'bicubic')
img = M.asmatrix(img, dtype='float64')  # found out main source of issues!
org = deepcopy(img) # copy image to preserve original

fig = plt.gcf()
fig.suptitle('Logarithmic Quantization')

plt.subplot(231)
plt.title('original')
plt.imshow(org, I.Greys_r)

H.haar_forward(img) # apply forward transformation
plt.subplot(232)
plt.title('forward')
plt.imshow(img, I.Greys_r)

offset = 4
for idx, th in enumerate([0.8, 0.9, 0.95]):
    img_cp = deepcopy(img) # copy image to preserve original
    thresh, lmaxt = Q.log_thresh(img_cp, th) # compute log threshold
    img_cp = Q.encode(img_cp, thresh, lmaxt) # apply log quantization
    img_cp = Q.decode(img_cp, thresh, lmaxt) # convert back to float
    H.haar_inverse(img_cp) # apply inverse haar transform

    plt.subplot(eval('23%s' % (offset + idx)))
    plt.title('%s' % th)
    plt.imshow(img_cp, I.Greys_r) # plot recovered image

plt.show()
