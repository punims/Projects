"""
this is a file created by Edan Patt for the Image Processing course 2018.
"""

# Imports
from imageio import imread
from skimage.color import rgb2gray
import numpy as np
import random
from . import sol5_utils as utils
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from scipy.ndimage.filters import convolve


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
KERNEL_SIZE = 3

MIN_SIGMA = 0
MAX_SIGMA = 0.2
NOISE_HEIGHT = 24
NOISE_WIDTH = 24
NOISE_CHANNELS = 48
NOISE_BATCH = 100
NOISE_STEPS_PER_EPOCH = 100
NOISE_EPOCH = 5
NOISE_SAMPLE_SIZE = 1000

QUICK_NOISE_BATCH = 10
QUICK_NOISE_STEPS_PER_EPOCH = 3
QUICK_NOISE_EPOCH = 2
QUICK_NOISE_SAMPLE_SIZE = 30

BLUR_HEIGHT = 16
BLUR_WIDTH = 16
BLUR_CHANNELS = 32
BLUR_KERNEL_SIZE = 7
BLUR_BATCH = 100
BLUR_STEPS_PER_EPOCH = 100
BLUR_EPOCH = 10
BLUR_SAMPLE = 1000

QUICK_BLUR_BATCH = 10
QUICK_BLUR_STEPS_PER_EPOCH = 3
QUICK_BLUR_EPOCH = 2
QUICK_BLUR_SAMPLE = 30


# Global Variables
cache = dict()


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


def random_crop(im, crop_size):
    """
    takes an image returns a tuple of a random crop from it and the coordinates for the crop.
    :param im: an np image
    :param crop_size: size of the crop to be returned (tuple)
    the crop takes place with the coordinates acting as the upper left corner of the im
    :return: tuple of the cropped image and the coordinate which was taken to crop it
    """
    height = np.random.randint(0, im.shape[0] - crop_size[0])
    width = np.random.randint(0, im.shape[1] - crop_size[1])
    # limiting the crop to the boundaries of the image.
    coordinates = (height, width)
    return im[height: height + crop_size[0], width: width + crop_size[1]], coordinates


def get_patch(original_im, corruption_func, crop_size):
    """
    helper function to get patches from an image
    Takes an image and then takes a crop of that picture and corrupts it
    :return: a tuple of cropped sized images, one from the original picture and the other
    which is corrupted.
    """
    first_crop, dump = random_crop(original_im, (crop_size[0]*3, crop_size[1]*3))
    corrupted_crop = corruption_func(first_crop)
    final_crop, coordinates = random_crop(first_crop, crop_size)
    return (final_crop.reshape(crop_size[0], crop_size[1], 1) - 0.5, corrupted_crop[coordinates[0]: coordinates[0] + crop_size[0],
                        coordinates[1]: coordinates[1] + crop_size[1]].reshape(crop_size[0], crop_size[1], 1) - 0.5)

    # reshape and reduce pixel values by 0.5


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    A generator function which yields a tuple of s random tuples of the form
    (source_batch, target_batch), where each output variable is an array of shape (batch_size,
    height, width, 1), target_batch is made of clean images, and source_batch is their respective
    randomly corrupted version according to corruption_func(im)
    :param filenames: – A list of filenames of clean images
    :param batch_size: The size of the batch of images for each iteration of
    Stochastic Gradient Descent.
    :param corruption_func: A function receiving a numpy’s array representation of an image as a
     single argument, and returns a randomly corrupted version of the input image.
    :param crop_size: A tuple (height, width) specifying the crop size of the patches to extract
    :return:
    """

    while True:
        i = 0
        normal_list = np.zeros((batch_size, crop_size[0], crop_size[1], 1))
        corrupted_list = np.zeros((batch_size, crop_size[0], crop_size[1], 1))
        while i < batch_size:
            im = random.choice(filenames) # im is the path which is a key in the cache
            if im not in cache.keys():
                cache[im] = read_image(im, GRAY)
            normal_im, corrupted_im = get_patch(cache[im], corruption_func, crop_size)
            normal_list[i] = normal_im
            corrupted_list[i] = corrupted_im
            i = i + 1
        yield (corrupted_list, normal_list)


def resblock(input_tensor, num_channels):
    """
    takes as input a symbolic input tensor and the number of channels for each of its
    convolutional layers, and returns a resblock according to the Resnet architecture
    :param input_tensor: a symbolic input tensor
    :param num_channels: the number of channels for the convolution
    :return resblock of shape Input -> Conv2D -> Relu -> Conv2D -> Add(Input, O) -> Relu
    """
    block = Conv2D(num_channels, (3, 3), padding='same')(input_tensor)
    block = Activation('relu')(block)
    block = Conv2D(num_channels, (3, 3), padding='same')(block)
    block = Add()([input_tensor, block])  # at this point block is O
    block = Activation('relu')(block)
    return block


def build_nn_model(height, width, num_channels, num_res_blocks):
    """

    :param height: Dimension for the symbolic input
    :param width: Dimension for the symbolic input
    :param num_channels: number of channels for the convolutions
    :param num_res_blocks: number of res blocks in the model.
    :return: return an untrained Keras model
    """
    input_tensor = Input(shape=(height, width, 1))
    block = input_tensor
    block = Conv2D(num_channels, (3, 3), padding='same')(block)
    block = Activation('relu')(block)
    for i in range(num_res_blocks):
        block = resblock(block, num_channels)
    block = Conv2D(1, (3, 3), padding='same')(block)  # final convolution has 1 output channel.
    final = Add()([input_tensor, block])
    model = Model(inputs=input_tensor, outputs=final)
    return model


def train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs,
                num_valid_samples):

    """
    divide the images into a training set and validation set, using an 80-20 split
    and generate from each set a dataset
    We then compile the models and train the models.
    :param model:  general neural network model for image restoration.
    :param images: – a list of file paths pointing to image files.
    :param corruption_func: same as described in section 3.
    :param batch_size: the size of the batch of examples for each iteration of SGD.
    :param steps_per_epoch: The number of update steps in each epoch
    :param num_epochs: The number of epochs for which the optimization will run
    :param num_valid_samples: The number of samples in the validation set to test on after the epoch
    :return:
    """

    randomized_images = np.random.permutation(images)
    training_set = randomized_images[round(len(images) * 0.2):]  # 80% of the images in this set
    test_set = randomized_images[: round(len(images) * 0.2)]  # 20% of images in this set
    training_data = load_dataset(training_set, batch_size, corruption_func, (model.input_shape[1], model.input_shape[2]))
    test_data = load_dataset(test_set, batch_size, corruption_func, (model.input_shape[1], model.input_shape[2]))
    model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=0.9))
    model.fit_generator(training_data, steps_per_epoch=steps_per_epoch, epochs=num_epochs, validation_data=test_data, validation_steps=num_valid_samples)


def restore_image(corrupted_image, base_model):
    """
    :param corrupted_image: a grayscale image of shape (height, width) and with values in the
     [0, 1] range of type float64, You can assume the size of the image is at least as large as
     the size of the image patches during training
    :param base_model: a neural network trained to restore small patches
    :return: the restored image.
    """
    a = Input(shape=(corrupted_image.shape[0], corrupted_image.shape[1], 1))
    b = base_model(a)
    new_model = Model(inputs=a, outputs=b)
    updated_corrupted_image = corrupted_image.reshape(corrupted_image.shape[0], corrupted_image.shape[1], 1)
    #  must respect that our model expects a tensor with 4 dimensions instead of 3.
    im = new_model.predict(np.expand_dims(updated_corrupted_image - 0.5, axis=0), batch_size=1)[0]
    return np.clip(np.squeeze((im + 0.5), axis=2), 0, 1).astype('float64')


def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    adds random gaussian noise to an image
    :param image: an image
    :param min_sigma: min noise to be added in the range
    :param max_sigma: max noise to be added  in the range
    :return: image with an added noise
    """
    rand_sigma = np.random.uniform(min_sigma, max_sigma)
    im_altered = image + np.random.normal(loc=0, scale=rand_sigma, size=image.shape)
    im_altered *= NORMALIZE
    im_altered = np.round(im_altered) / NORMALIZE
    return np.clip(im_altered, 0, 1)


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    """
    trains a denoising model
    :param num_res_blocks: number of res blocks for the model
    :param quick_mode:
    :return:
    """
    images = utils.images_for_denoising()
    if quick_mode:
        model = build_nn_model(NOISE_HEIGHT, NOISE_WIDTH, NOISE_CHANNELS, num_res_blocks)
        train_model(model, images, corruption_function, QUICK_NOISE_BATCH, QUICK_NOISE_STEPS_PER_EPOCH, QUICK_NOISE_EPOCH, QUICK_NOISE_SAMPLE_SIZE)
    else:
        model = build_nn_model(NOISE_HEIGHT, NOISE_WIDTH, NOISE_CHANNELS, num_res_blocks)
        train_model(model, images, corruption_function, NOISE_BATCH, NOISE_STEPS_PER_EPOCH, NOISE_EPOCH, NOISE_SAMPLE_SIZE)
    return model


def corruption_function(im):
    """
    return a premade corruption function
    :param im:
    :return:
    """
    return add_gaussian_noise(im, MIN_SIGMA, MAX_SIGMA)


def blur_function(im):
    """
    return the blur function with size kernel of 7
    :param im:
    :return:
    """
    return random_motion_blur(im, [BLUR_KERNEL_SIZE])


def add_motion_blur(image,kernel_size,angle):
    """
    simulates a blur using a square kernel of size kernel size with the given angle
    :param image: the image to be blurred
    :param kernel_size: size of the square blur kernel
    :param angle: the angle at which the kernel has a diagonal line
    :return: blurred image
    """
    blur_kernel = utils.motion_blur_kernel(kernel_size, angle)
    return convolve(image, blur_kernel)


def random_motion_blur(image,list_of_kernel_sizes):
    """
    samples in a uniform manner an angle between 0 and Pi  and chooses a kernel size at random
    from the list provided
    returns the image after adding motion blur while being normalized and clipped
    :param image: the image to be blurred
    :param list_of_kernel_sizes: a kernel size will be picked at random from here
    :return: blurred and normalized image
    """
    angle = np.random.uniform(0, np.pi)
    kernel_size = np.random.choice(list_of_kernel_sizes)
    im = add_motion_blur(image, kernel_size, angle)
    im *= NORMALIZE
    im = np.round(im) / NORMALIZE
    return np.clip(im, 0, 1)


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    """
    creates model that learns how to deblur an image
    :param num_res_blocks: number of res blocks
    :param quick_mode: for different modes
    :return: trained model for deblurring
    """
    images = utils.images_for_deblurring()
    if quick_mode:
        model = build_nn_model(BLUR_HEIGHT, BLUR_WIDTH, BLUR_CHANNELS, num_res_blocks)
        train_model(model, images, blur_function, QUICK_BLUR_BATCH, QUICK_BLUR_STEPS_PER_EPOCH, QUICK_BLUR_EPOCH, QUICK_BLUR_SAMPLE)
    else:
        model = build_nn_model(BLUR_HEIGHT, BLUR_WIDTH, BLUR_CHANNELS, num_res_blocks)
        train_model(model, images, blur_function, BLUR_BATCH, BLUR_STEPS_PER_EPOCH, BLUR_EPOCH, BLUR_SAMPLE)
    return model