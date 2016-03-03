# image load, store, display
from matplotlib.image import imread
from scipy.misc import imresize


def rgb2gray(rgb):
    '''
    Convert rgb image to grayscale
    '''
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def read_gecko_image():
    """
    Read Gecko image as 256 x 256 grayscale matrix.
    """
    # load image
    img = imread('gecko.jpg')
    img = rgb2gray(img)
    img = imresize(img, [256, 256], 'bicubic')
    img = img.astype(dtype='float64')
    return img
