"""
this is a file created by Edan Patt for the Image Processing course 2018.
"""


# Imports
import math
import numpy as np
from imageio import imread, imwrite
from skimage.color import rgb2gray
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


# Constants:
GRAY = 1
RGB = 2
GRAY_DIM = 2
RGB_DIM = 3
NORMALIZE = 255
COLUMN = 1
ROW = 0



def read_image(filename, representation):
    """
    This function returns an image, the output image is represented by a matrix of type
    np.float64 with intensities (either grayscale or RGB channel intensities) normalized to the
    range [0, 1].
    :param filename: string containing the image filename to read
    :param representation: representation code, either 1 or 2 defining whether the output should be
    a grayscale image (1) or an RGB image (2).
    """

    # first normalize the image.
    image = imread(filename)
    image = image.astype(np.float64)
    image_float = image / NORMALIZE
    if representation == GRAY and len(image.shape) == RGB_DIM:  # turn RGB image to grayscale
        gray_image = rgb2gray(image_float)
        return gray_image
    elif representation == RGB:  # return image as is (after normalization)
        return image_float
    elif representation == GRAY and len(image.shape) == GRAY_DIM:
        return image_float
    else:  # representation != 1 or 2
        raise ValueError("wrong value passed to representation, please try '1' or '2'")


def columnDFT(signal):
    """
    :param signal: signal is an array of dtype float64 with shape (N,1)
    go over a column and return a column dft.
    :return:
    """
    N = signal.shape[ROW]
    x = np.arange(signal.shape[ROW])
    u = x.reshape([N, 1])
    omega = -2j * np.pi * x * u / N  # omega will be an NxN matrix.
    complex_fourier = np.dot(np.exp(omega), signal)  # NxN dot Nx1 = Nx1
    return complex_fourier

def columnIDFT(fourier_signal):
    """
    helper function for IDFT
    :param fourier_signal: complex array
    :return: real column
    """
    N = fourier_signal.shape[0]
    x = np.arange(fourier_signal.shape[0])
    u = x.reshape([N, 1])
    omega = 2j * np.pi * u * x / N
    signal = np.dot(np.exp(omega), fourier_signal) / N
    return signal

def vanderMondeMatrix(size):
    """
    returns vandermonde matrix of correct size.
    :param size:
    :return:
    """

    omega = np.exp(-2j * np.pi * np.arange(size) / size).reshape(-1, 1)
    return omega ** np.arange(size)


def DFT(signal):
    """
    :param signal: signal is an array of dtype float64 with shape (N,1)
    :return: complex Fourier signal and complex signal, respectively. Note that when the origin of
    fourier_signalis a transformed real signal you can expect signal to be real valued as well,
    although it may return wit a tiny imaginary part which may be ignored.
    """

    if signal.shape[COLUMN] == 1:  # single array has N rows and one column
        return columnDFT(signal)
    else:  # dealing with a larger matrix for DFT2 (bonus)
        vander_matrix = vanderMondeMatrix(signal.shape[0])
        vander_rows = np.dot(vander_matrix, signal).T
        vander_matrix_2 = vanderMondeMatrix(vander_rows.shape[0])
        return np.dot(vander_matrix_2, vander_rows).T



def IDFT(fourier_signal):
    """
    :param fourier_signal: fourier_signal
    is an array of dtype complex128 with the same shape Nx1
    :return: complex signal
    """
    if fourier_signal.shape[COLUMN] == 1:  # single array has N rows and one column
        return columnIDFT(fourier_signal)
    else:  # dealing with a larger matrix for DFT2 (bonus)
        vander_inverse = np.linalg.inv(vanderMondeMatrix(fourier_signal.shape[0]))
        vander_rows = np.dot(vander_inverse, fourier_signal).T
        vander_inverse_2 = np.linalg.inv(vanderMondeMatrix(vander_rows.shape[0]))
        return np.dot(vander_inverse_2, vander_rows).T

def DFT2(image):
    """
    :param image: image is a grayscale image of dtype float64
    :return: fourier transformed image.
    """
    return DFT(image)

def IDFT2(fourier_image):
    """
    :param fourier_image: fourier_image is a 2D array of dtype complex128
    :return:
    """
    return IDFT(fourier_image)


def conv_der(im):
    """
    the input and the output are grayscale images of type float64, and the output is the magnitude
    of the derivative, with the same dtype and shape. The output should be calculated in the following way:
    magnitude = np.sqrt (np.abs(dx)**2 + np.abs(dy)**2)
    """

    derivative_array = np.array([1, 0, -1]).reshape(1, 3)
    dx = convolve2d(im, derivative_array, mode='same')
    dy = convolve2d(im, derivative_array.T, mode='same')
    magnitude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)
    return magnitude


def derive_fourier(fourier_image, type):
    """
    helping function to compute x or y derivative
    :param fourier_image:
    :return: derivative according to type.
    """
    fourier_shift = np.fft.fftshift(fourier_image)  # shift F(0,0) to middle
    if type == 'x':
        N = fourier_image.shape[ROW]
        u = np.arange(np.floor(-N/2), np.floor(N/2)).reshape(N, 1)
        # 1D array used to multiply by index
        return np.fft.ifftshift(2j * np.pi * u / N * fourier_shift)
    elif type == 'y':
        N = fourier_image.shape[COLUMN]
        v = np.arange(np.floor(-N/2), np.floor(N/2)).reshape(1, N)
        return np.fft.ifftshift(2j * np.pi * v / N * fourier_shift)

def fourier_der(im):
    """

    :param im: float64 grayscale image
    :return: float64 grayscale image of magnitude using forier transform to calculate the
    derivatives
    """
    fourier_im = DFT2(im)
    dx = derive_fourier(fourier_im, 'x')
    inverse_dx = IDFT2(dx)  # must be inverted according to the algorithm
    dy = derive_fourier(fourier_im, 'y')
    inverse_dy = IDFT2(dy)
    magnitude = np.sqrt(np.abs(inverse_dx) ** 2 + np.abs(inverse_dy) ** 2)
    # same magnitude as before
    return magnitude


def create_guassian(kernel_size):
    """
    helping function to create guassian kernal
    :param kernal_size:
    :return: np.array of size kernal_size X kernal_size
    """

    if kernel_size % 2 == 0:
        raise ValueError("wrong value passed to guassian blur, please pass an odd kernal size")
    origin = np.array([0.5, 0.5]).reshape(1, 2)
    # we start with a 0.5, 0.5 instead of 1,1 to keep the sum to 1.
    seed = origin
    i = 0
    while i < kernel_size - 2:  # requires -2 loops for the correct size.
        seed = convolve2d(seed, origin)
        i += 1
    return convolve2d(seed, seed.T)  # return convolution of seed with his transpose for the matrix


def blur_spatial (im, kernel_size):
    """
    :param im: is the input image to be blurred (grayscale float64 image)
    :param kernel_size: - is the size of the gaussian kernel in each dimension (an odd integer).
    :return: The function returns the output blurry image (grayscale float64 image)
    """

    guassian_kernel = create_guassian(kernel_size)
    return convolve2d(im, guassian_kernel, mode='same')


def crop_center(img, guasian_kernel, cropx, cropy):
    """
    helping function which places guassian kernel in the middle of a padded array
    :param img:
    :param cropx:
    :param cropy:
    :return:
    """
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    img[starty:starty+cropy,startx:startx+cropx] = guasian_kernel
    return img

def blur_fourier (im, kernel_size):
    """
    blur the image in the fourier space.
    :param im: grayscale image
    :param kernel_size: odd kernal size
    :return: blurred original image, grayscale float64
    """

    guassian_kernal = create_guassian(kernel_size)
    g = np.zeros(im.shape)
    g = crop_center(g, guassian_kernal, guassian_kernal.shape[ROW], guassian_kernal.shape[COLUMN])
    g = np.fft.ifftshift(g)  # guassian kernal center placed at 0,0
    F = DFT2(im)
    G = DFT2(g)
    return IDFT2(F*G).real
