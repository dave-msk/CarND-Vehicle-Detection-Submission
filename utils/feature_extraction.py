import numpy as np
import cv2
from skimage.feature import hog


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    """
    Get HOG features of an image
    
    :param img: input image
    :param orient: number of orientations
    :param pix_per_cell: number of pixels in each cell
    :param cell_per_block: number of cells in each block
    :param vis: visualization flag
    :param feature_vec: flag for auto flattening of hog features
    :return: hog_feature[, visualization]
    """
    return hog(img, orientations=orient,
               pixels_per_cell=(pix_per_cell, pix_per_cell),
               cells_per_block=(cell_per_block, cell_per_block),
               transform_sqrt=False,
               visualise=vis, feature_vector=feature_vec)


def bin_spatial(img, size=(32, 32)):
    """
    Get spatial histogram feature
    
    :param img: input image
    :param size: size to extract histogram
    :return: concatenated spatial feature data of all channels
    """
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32, bins_range=(0, 256)):
    """
    Get color histogram features
    
    :param img: input image
    :param nbins: number of bins
    :param bins_range: range of bins
    :return: vector of color histogram features
    """
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the feature vector
    return hist_features

