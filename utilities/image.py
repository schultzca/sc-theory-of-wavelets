# image load, store, display
from matplotlib.image import imread
from matplotlib.cm import Greys_r
from scipy.misc import imresize

def rgb2gray(rgb):
    '''
    Convert rgb image to grayscale
    '''
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
