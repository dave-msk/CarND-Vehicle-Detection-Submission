import numpy as np
import itertools
import random
import cv2


# Dictionary for color space parsing.
color_spaces = {'HLS': cv2.COLOR_RGB2HLS,
                'HSV': cv2.COLOR_RGB2HSV,
                'YUV': cv2.COLOR_RGB2YUV,
                'LUV': cv2.COLOR_RGB2LUV,
                'YCrCb': cv2.COLOR_RGB2YCrCb}


def convert_color(image, color_space='RGB'):
    """
    Convert image to specified color space
    
    :param image: input image
    :param color_space: target color space, supported color spaces are:
           {'HLS', 'HSV', 'YUV', 'LUV', 'YCrCb'}.
    :return: image array in specified color space representation,
             returns the same image if `color_space` is other than the above 5
    """
    global color_spaces
    cvt = color_spaces.get(color_space, None)
    if cvt:
        img = cv2.cvtColor(image, cvt)
    else:
        img = np.copy(image)
    return img


def balance_lists(data_list1, data_list2):
    """
    Balance the number of data by duplicating samples in the smaller list
    inplace randomly.
    """
    less, more = (data_list1, data_list2) if len(data_list1) < len(data_list2)\
                                        else (data_list2, data_list1)
    old_less = list(less)
    while len(less) < len(more):
        diff = len(more) - len(less)
        random.shuffle(old_less)
        less.extend(old_less[:diff])


def gen_binary_labels(distrib):
    """
    Generate binary labels
    
    :param distrib: 2-tuple in the form (noncars_num, cars_num)
    """
    return np.concatenate((np.zeros(distrib[0]), np.ones(distrib[1])))


def get_sliding_windows(x_start_stop, y_start_stop, xy_window_width, xy_pix_per_step):
    """
    Generate sliding window boxes
    
    :param x_start_stop: x-range - (xstart, xstop)
    :param y_start_stop: y-range - (ystart, ystop)
    :param xy_window_width: length of window sides - (width, height)
    :param xy_pix_per_step: number of pixel per step - (xstep, ystep)
    :return: list of pairs of top-left and bottom right points of each box
    """
    windows_list = []
    x_range = range(x_start_stop[0], x_start_stop[1] - xy_window_width[0] + 1, xy_pix_per_step[0])
    y_range = range(y_start_stop[0], y_start_stop[1] - xy_window_width[1] + 1, xy_pix_per_step[1])
    for x, y in itertools.product(x_range, y_range):
        windows_list.append(((x, y), (x+xy_window_width[0], y+xy_window_width[1])))
    return windows_list



