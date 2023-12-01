from pylab import imshow, show, imsave
import matplotlib as plt
import mahotas
import mahotas.demos
import numpy as np


def find_bounding_box(mask):
    # Find the indices of True values
    y_indices, x_indices = np.where(mask)

    if not len(x_indices) or not len(y_indices):
        return None  # No True values in the mask

    # Find min and max of x and y indices
    min_x, max_x = min(x_indices), max(x_indices)
    min_y, max_y = min(y_indices), max(y_indices)

    return (min_x, min_y, max_x, max_y)
def run_edge_detector(image_path, ind):
    # loading the image
    # wally = mahotas.demos.load('wally')
    wally = mahotas.imread(image_path, as_grey=None)

    # getting float type value
    # float values are better to use
    wfloat = wally.astype(float)

    # splitting image into red, green and blue channel
    r, g, b = wfloat.transpose((2, 0, 1))

    # white channel
    w = wfloat.mean(2)

    # pattern of wally shirt
    # pattern + 1, +1, -1, -1 on vertical axis
    pattern = np.ones((24, 16), float)
    for i in range(2):
        pattern[i::4] = -1

    # convolve with the red minus white
    # increase the response where shirt is
    v = mahotas.convolve(r-w, pattern)

    # getting maximum value
    mask = (v == v.max())

    # creating mask to tone down the image
    # except the region where wally is
    mask = mahotas.dilate(mask, np.ones((48, 24)))

    # subtraction mask from the wally
    np.subtract(wally, .8 * wally * ~mask[:, :, None],
                       out = wally, casting ='unsafe')

    # show the new image
    imsave(f'result/edge_detection/edge_detection_result_{ind}.jpg', wally)

    return find_bounding_box(mask)