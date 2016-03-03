import utilities.wavelet as wavelet
import utilities.quantization as quantization
import utilities.image as I
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from copy import deepcopy
from math import sqrt

img = I.read_gecko_image()
org = deepcopy(img)

fig = plt.gcf()
fig.suptitle('Logarithmic quantizationuantization')

plt.subplot(231)
plt.title('original')
plt.imshow(org, cm.Greys_r)

haar_coefficients = [1/sqrt(2), 1/sqrt(2)]

wavelet.forward(h=haar_coefficients, image=img)
plt.subplot(232)
plt.title('forward')
plt.imshow(img, cm.Greys_r)

offset = 4
for idx, th in enumerate([0.8, 0.9, 0.95]):
    img_cp = deepcopy(img)
    thresh, lmaxt = quantization.log_thresh(img_cp, th)
    img_cp = quantization.encode(img_cp, thresh, lmaxt)
    img_cp = quantization.decode(img_cp, thresh, lmaxt)
    wavelet.inverse(h=haar_coefficients, image=img_cp)

    plt.subplot(eval('23%s' % (offset + idx)))
    plt.title('%s' % th)
    plt.imshow(img_cp, cm.Greys_r)

plt.show()
