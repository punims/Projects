"""
this is a file created by Edan Patt for the Image Processing course 2018.
"""

# Imports
import math
import numpy as np
from imageio import imread, imwrite
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage.filters import convolve
import os


from skimage.transform import pyramid_gaussian
import cv2


# Constants:
GRAY = 1
RGB = 2
GRAY_DIM = 2
RGB_DIM = 3
NORMALIZE = 255
MIN_SIZE = 32
COLUMN = 1
ROW = 0
START = 1


def relpath(filename):
    """
    returns relative path to a file
    """
    return os.path.join(os.path.dirname(__file__),filename)


def create_guassian(filter_size):
    """
    helping function to create gaussian kernel
    :param filter_size: odd scalar for blurring.
    :return: np.array of size 1 x kernel_size
    """

    if filter_size % 2 == 0:
        raise ValueError("wrong value passed to please pass an odd  size")
    origin = np.array([0.5, 0.5]).reshape(1, 2)
    # we start with a 0.5, 0.5 instead of 1,1 to keep the sum to 1.
    seed = origin
    i = 0
    while i < filter_size - 2:  # requires -2 loops for the correct size.
        seed = convolve2d(seed, origin)
        i += 1
    return seed


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


def image_shrink(curr_image, filter_vec):
    """
    return an image that is half as large on both dimensions
    :param curr_image: the current image
    :param filter_vec: the convolution kernel.
    :return: shrunken image, first blurred then sampled.
    """
    im1 = convolve(curr_image, filter_vec)
    im2 =  convolve(im1.T, filter_vec).T
    return im2[::2, ::2]


def image_expand(curr_image, filter_vec):
    """
    returns an expanded image by padding the current image with zeroes and then blurring.
    :param curr_image: the current picture
    :param filter_vec: convolution kernel
    :return: padded image after being blurred
    """
    added_zeros = np.zeros((curr_image.shape[ROW] * 2, curr_image.shape[COLUMN] * 2))
    added_zeros[1::2, ::2] = curr_image  # put the image in the odd places of the zero matrix.
    filter_vec = filter_vec * 2
    return convolve(convolve(added_zeros, filter_vec), filter_vec.T)
    # multiply by 2 because the average of pixels is now 3/4 added 0s. normalization must be changed.


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    builds a Gaussian pyramid
    :param im:  im – a grayscale image with double values in [0, 1] (e.g. the output of ex1’s
    read_image with the representation
    set to 1).
    :param max_levels: – the maximal number of levels1 in the resulting pyramid
    :param filter_size:  the size of the Gaussian filter (an odd scalar that represents a squared
    filter) to be used in constructing the pyramid filter (e.g for filter_size = 3 you should get
    [0.25, 0.5, 0.25])
    :return: tuple of pyr and filter_vec:
    pyr as a standard python array (i.e. notnumpy’s array) with maximum length of
    max_levels, where each element of the array is a grayscale image.
    filter_vec which is row vector of shape (1, filter_size) used
    for the pyramid construction. This filter should be built using a consequent 1D
    convolutions of [1 1] with itself in order to derive a row of the binomial coefficients
    which is a good approximation to the Gaussian profile. The filter_vec should be
    normalized.
    """

    pyr = []
    pyr.append(im)
    i = 0
    curr_image = im.copy()
    filter_vec = create_guassian(filter_size)
    while i < max_levels - 1 and (curr_image.shape[ROW]>=MIN_SIZE and curr_image.shape[COLUMN]>=MIN_SIZE):
        curr_image = image_shrink(curr_image, filter_vec)
        pyr.append(curr_image)
        i += 1
    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Builds a Laplacian Pyramid
    :param im:  im – a grayscale image with double values in [0, 1] (e.g. the output of ex1’s
    read_image with the representation
    set to 1).
    :param max_levels: – the maximal number of levels1 in the resulting pyramid
    :param filter_size:  the size of the Gaussian filter (an odd scalar that represents a squared
    filter) to be used in constructing the pyramid filter (e.g for filter_size = 3 you should get
    [0.25, 0.5, 0.25])
    :return: tuple of pyr and filter_vec:
    pyr as a standard python array (i.e. notnumpy’s array) with maximum length of
    max_levels, where each element of the array is a grayscale image.
    filter_vec which is row vector of shape (1, filter_size) used
    for the pyramid construction. This filter should be built using a consequent 1D
    convolutions of [1 1] with itself in order to derive a row of the binomial coefficients
    which is a good approximation to the Gaussian profile. The filter_vec should be
    normalized.
    """

    pyr = []
    gaus_pyramid, kernel = build_gaussian_pyramid(im, max_levels, filter_size)
    for i in range(len(gaus_pyramid) - 1):  # gaus_pyramid <= max_levels
        pyr.append(gaus_pyramid[i] - image_expand(gaus_pyramid[i+1], kernel))
    pyr.append(gaus_pyramid[-1])
    return pyr, kernel


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    :param lpyr: Laplacian pyramid
    :param filter_vec: gaussian filter generated in build laplacian
    :param coeff: is a python list. The list length is the same as the number of levels in the
    pyramid lpyr. Before reconstructing the image img you should multiply each level i of the
    laplacian pyramid by its corresponding coefficient coeff[i].
    :return: the original image that originated the laplacian pyramid
    """

    if len(coeff) != len(lpyr):
        raise ValueError("coeff and pyramid lengths do not match, please try again")
    for i in range(len(coeff)):
        lpyr[i] = lpyr[i] * coeff[i]  # multiply by the coefficient
    im = lpyr[-1]
    for i in range(len(lpyr) - 1, 0, -1):
        im = (lpyr[i-1] + image_expand(im, filter_vec))
    return im


def stretch_im(image):
    """
    stretches image
    :param image:
    :return: stretched image
    """
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def render_pyramid(pyr, levels):
    """
    :param pyr: Either a Gaussian or a Laplacian pyramid
    :param levels: the number of levels to present
    :return: a single black image in which the pyramid levels are stacked horizontally.
    """

    if len(pyr) < levels or levels <= 0:
        raise ValueError("invalid levels given pyramid, try again")
    im = stretch_im(pyr[0])
    for i in range(START, levels):
        difference = pyr[0].shape[ROW] - pyr[i].shape[ROW]
        im = np.hstack((im, np.pad(stretch_im(pyr[i]), ((0, difference), (0,0)), 'constant')))
    return im


def display_pyramid(pyr, levels):
    """
    displays a pyramid horizontally from largest to smallest image, background is black.
    :param pyr: the pyramid
    :param levels: the amount of levels displayed
    """
    image = render_pyramid(pyr, levels)
    plt.imshow(image, cmap="gray")
    plt.show()


def create_blend_pyramid(l1, l2, gm):
    """
    helping function to create blended image pyramid
    :param l1: laplacian of im1
    :param l2: laplacian of im2
    :param gm: gaussian of mask
    :param max_levels: max levels in the pyramid
    :return: blended pyramid
    """

    blend_pyramid = []
    for i in range(len(l1)):
        lvlkimage = gm[i] * l1[i] + (1 - gm[i]) * l2[i]
        blend_pyramid.append(lvlkimage)
    return blend_pyramid


def pyramid_blending(im1,im2,mask,max_levels,filter_size_im,filter_size_mask):
    """

    :param im1, im2: are two input grayscale images to be blended
    :param mask: a boolean mask containing true and false values in the corresponding i,j coordinates
                  1 equating to truth and 0 to false.
    :param max_levels: amount of levels for the pyramids
    :param filter_size_im: size of the gaussian filter for the laplacian pyramids of both im1 and im2
    :param filter_size_mask: size of the gaussian filter for the gaussian pyramid for the mask
    :return: blended image
    """

    l1, trash1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    l2, trash2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    gm, vector = build_gaussian_pyramid(mask.astype('float64'), max_levels, filter_size_mask)
    #  mask is bool, must be used as a float64 picture
    blend_pyramid = create_blend_pyramid(l1, l2, gm)
    coeff = [1] * len(l1)
    return laplacian_to_image(blend_pyramid, vector, coeff)


def color_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    does pyramid blending for each channel of color.
    :param im1: color images
    :param mask: boolean mask
    :param max_levels: amount of levels for the pyramid
    :param filter_size_im: filter for the image
    :param filter_size_mask: filter for the mask
    :return: blended color image.
    """
    #  build pyramid for each color channel separately
    r = pyramid_blending(im1[:, :, 0], im2[:, :, 0], mask, max_levels, filter_size_im, filter_size_mask)
    g = pyramid_blending(im1[:, :, 1], im2[:, :, 1], mask, max_levels, filter_size_im, filter_size_mask)
    b = pyramid_blending(im1[:, :, 2], im2[:, :, 2], mask, max_levels, filter_size_im, filter_size_mask)
    blended_im = np.empty(im1.shape)
    blended_im[:, :, 0] = r
    blended_im[:, :, 1] = g
    blended_im[:, :, 2] = b
    #  add each blended color channel back in to one picture
    return blended_im


def show4images(im1, im2, mask, blended):
    """
    function that uses subplot to display 4 images in the same figure
    """
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(im1)
    plt.subplot(2,2,2)
    plt.imshow(im2)
    plt.subplot(2,2,3)
    plt.imshow(mask, cmap='gray')
    plt.subplot(2,2,4)
    plt.imshow(blended)
    plt.show()


def blending_example1():
    """
    example of a nice picture 1
    """
    im1 = read_image(relpath('externals/galaxy.jpg'), 2)
    im2 = read_image(relpath('externals/westwall.jpg'), 2)
    mask = read_image(relpath('externals/mask1.jpg'), 1)
    mask = mask > 0.5  # make mask boolean.
    blended = np.clip(color_blending(im1, im2, mask, 3, 51, 3), 0, 1)
    show4images(im1, im2, mask, blended)
    return im1, im2, mask.astype('bool'), blended


def blending_example2():
    """
    example of a nice picture 2
    """
    im1 = read_image(relpath('externals/foreveralone.jpg'), 2)
    im2 = read_image(relpath('externals/fullmoon.jpg'), 2)
    mask = read_image(relpath('externals/mask2.jpg'), 1)
    mask = mask > 0.5  # make mask boolean.
    blended = np.clip(color_blending(im1, im2, mask, 5, 21, 3), 0, 1)
    show4images(im1, im2, mask, blended)
    return im1, im2, mask.astype('bool'), blended


blending_example2()