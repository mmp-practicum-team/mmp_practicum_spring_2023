import random
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np


def generate_random_data(height, width, count):
    """
    Generates N pairs of image/masks with provided height and width
    :param height: height pixels int
    :param width: width pixels int
    :param count: image pairs count int
    :return: array of generated images
    """
    x, y = zip(*[generate_img_and_mask(height, width) for i in range(0, count)])

    X = np.asarray(x) * 255
    X = X.repeat(3, axis=1).transpose([0, 2, 3, 1]).astype(np.uint8)
    Y = np.asarray(y)

    return X, Y


def generate_img_and_mask(height, width):
    """
    Generates sigle image and mask with shapes on them
    :param height: height pixels int
    :param width: width pixels int
    :return: image, mask tuple
    """
    shape = (height, width)

    triangle_location = get_random_location(*shape)
    circle_location1 = get_random_location(*shape, zoom=0.7)
    circle_location2 = get_random_location(*shape, zoom=0.5)
    mesh_location = get_random_location(*shape)
    square_location = get_random_location(*shape, zoom=0.8)
    plus_location = get_random_location(*shape, zoom=1.2)

    # Create input image
    arr = np.zeros(shape, dtype=bool)
    arr = add_triangle(arr, *triangle_location)
    arr = add_circle(arr, *circle_location1)
    arr = add_circle(arr, *circle_location2, fill=True)
    arr = add_mesh_square(arr, *mesh_location)
    arr = add_filled_square(arr, *square_location)
    arr = add_plus(arr, *plus_location)
    arr = np.reshape(arr, (1, height, width)).astype(np.float32)

    # Create target masks
    masks = np.asarray([
        add_filled_square(np.zeros(shape, dtype=bool), *square_location),
        add_circle(np.zeros(shape, dtype=bool), *circle_location2, fill=True),
        add_triangle(np.zeros(shape, dtype=bool), *triangle_location),
        add_circle(np.zeros(shape, dtype=bool), *circle_location1),
        add_filled_square(np.zeros(shape, dtype=bool), *mesh_location),
        # add_mesh_square(np.zeros(shape, dtype=bool), *mesh_location),
        add_plus(np.zeros(shape, dtype=bool), *plus_location)
    ]).astype(np.float32)

    return arr, masks


def add_square(arr, x, y, size):
    """
    Adds hollow square to image at given position with given size
    :param arr: numpy array image
    :param x: position x int
    :param y: position y int
    :param size: square side size
    :return: Modified array
    """
    s = int(size / 2)
    arr[x - s, y - s:y + s] = True
    arr[x + s, y - s:y + s] = True
    arr[x - s:x + s, y - s] = True
    arr[x - s:x + s, y + s] = True

    return arr


def add_filled_square(arr, x, y, size):
    """
    Adds filled square to image at given position with given size
    :param arr: numpy array image
    :param x: position x int
    :param y: position y int
    :param size: square side size
    :return: Modified array
    """
    s = int(size / 2)

    xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]

    return np.logical_or(arr, logical_and([xx > x - s, xx < x + s, yy > y - s, yy < y + s]))


def logical_and(arrays):
    """
    Logical AND between two arrays
    :param arrays: iterable of arrays
    :return: result array
    """
    new_array = np.ones(arrays[0].shape, dtype=bool)

    for a in arrays:
        new_array = np.logical_and(new_array, a)

    return new_array


def add_mesh_square(arr, x, y, size):
    """
    Adds mesh square to image at given position with given size
    :param arr: numpy array image
    :param x: position x int
    :param y: position y int
    :param size: square side size
    :return: Modified array
    """

    s = int(size / 2)

    xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]

    return np.logical_or(arr, logical_and([xx > x - s, xx < x + s, xx % 2 == 1, yy > y - s, yy < y + s, yy % 2 == 1]))


def add_triangle(arr, x, y, size):
    """
    Adds mesh triangle to image at given position with given size
    :param arr: numpy array image
    :param x: position x int
    :param y: position y int
    :param size: triangle side size
    :return: Modified array
    """

    s = int(size / 2)

    triangle = np.tril(np.ones((size, size), dtype=bool))

    arr[x - s:x - s + triangle.shape[0], y - s:y - s + triangle.shape[1]] = triangle

    return arr


def add_circle(arr, x, y, size, fill=False):
    """
    Adds filled circle to image at given position with given size
    :param arr: numpy array image
    :param x: position x int
    :param y: position y int
    :param size: circle side size
    :return: Modified array
    """

    xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]
    circle = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    new_arr = np.logical_or(arr, np.logical_and(circle < size, circle >= size * 0.7 if not fill else True))

    return new_arr


def add_plus(arr, x, y, size):
    """
    Adds plus to image at given position with given size
    :param arr: numpy array image
    :param x: position x int
    :param y: position y int
    :param size: circle side size
    :return: Modified array
    """

    s = int(size / 2)
    arr[x - 1:x + 1, y - s:y + s] = True
    arr[x - s:x + s, y - 1:y + 1] = True

    return arr


def get_random_location(width, height, zoom=1.0):
    """
    Returns random location for random coordinate
    :param width: image width pixels int
    :param height: image height pixels int
    :param zoom: zoom into image float
    :return:
    """
    x = int(width * random.uniform(0.1, 0.9))
    y = int(height * random.uniform(0.1, 0.9))

    size = int(min(width, height) * random.uniform(0.06, 0.12) * zoom)

    return x, y, size


def plot_img_array(img_array, ncol=3):
    """
    Plots image array
    :param img_array: image array numpy
    :param ncol: number of columns
    :return: None
    """
    nrow = len(img_array) // ncol

    fig, axes = plt.subplots(nrow, ncol, sharex='all', sharey='all', figsize=(ncol * 2, nrow * 2))

    for ax, image in zip(axes.reshape(-1), img_array):
        ax.imshow(image)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    fig.tight_layout()
    plt.show()


def plot_side_by_side(img_arrays):
    """
    Plot images side by side
    :param img_arrays: image arrays
    :return: None
    """
    flatten_list = reduce(lambda x, y: x + y, zip(*img_arrays))

    plot_img_array(flatten_list, ncol=len(img_arrays))


def masks_to_colorimg(masks):
    """
    Converts image mask to color image by class
    :param masks: mask array
    :return: converted array
    """
    colors = np.asarray([(101, 172, 228), (56, 34, 132), (160, 194, 56), (201, 58, 64), (242, 207, 1), (0, 152, 75)])

    colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255
    channels, height, width = masks.shape

    for y in range(height):
        for x in range(width):
            selected_colors = colors[masks[:, y, x] > 0.5]

            if len(selected_colors) > 0:
                colorimg[y, x, :] = np.mean(selected_colors, axis=0)

    return colorimg.astype(np.uint8)
