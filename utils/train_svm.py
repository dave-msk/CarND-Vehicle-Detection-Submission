from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import feature_extraction as fx
import utils as tu
import numpy as np
import pickle
import glob
import cv2
import sys
import os
import time

Cs = np.logspace(1e-4, 1e2, 7)
folds = 3
test_size = 0.2

color_space = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL".
spatial_size = (16, 16)  # Spatial binning dimensions, (32, 32)
hist_bins = 16    # Number of histogram bins, 32
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
window_side = 64


def extract_feature(image, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    Extract features of an image
    """
    # Create a list to contain different feature for later concatenation
    sub_features = []
    feature_image = tu.convert_color(image, color_space)

    # Extract spatial binning features
    if spatial_feat:
        spatial_features = fx.bin_spatial(feature_image, size=spatial_size)
        sub_features.append(spatial_features)

    # Extract color histogram features
    if hist_feat:
        hist_features = fx.color_hist(feature_image, nbins=hist_bins)
        sub_features.append(hist_features)

    # Extract HOG features
    if hog_feat:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(fx.get_hog_features(feature_image[:, :, channel],
                                                        orient, pix_per_cell, cell_per_block,
                                                        vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = fx.get_hog_features(feature_image[:, :, hog_channel], orient,
                                               pix_per_cell, cell_per_block,
                                               vis=False, feature_vec=True)
        sub_features.append(hog_features)
    return np.concatenate(sub_features)


def extract_features(image_filenames, resize=None):
    """
    Extract features given image file names
    
    :param image_filenames: list of image file names
    :param resize: resize dimension 
    :return: list of extracted features of corresponding image
    """
    features = []
    for name in image_filenames:
        image = cv2.imread(name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if resize:
            image = cv2.resize(image, resize)
        features.append(extract_feature(image, color_space, spatial_size,
                                        hist_bins, orient, pix_per_cell,
                                        cell_per_block, hog_channel,
                                        spatial_feat, hist_feat, hog_feat))
    return features


def train_model_cv(X_train, y_train, Cs, folds=3):
    """
    Train a linear SVM using k-fold cross-validation
    
    :param X_train: features 
    :param y_train: labels
    :param Cs: list of candidate parameter `C`
    :param folds: number of folds in k-fold cross-validation
    :return: trained linear SVM classifier
    """
    svc = LinearSVC()
    parameters = {'C': Cs}
    clf = GridSearchCV(svc, parameters, cv=folds)
    clf.fit(X_train, y_train)
    return clf


def save_model(data_file, clf, color_space, orient,
               pix_per_cell, cell_per_block,
               hog_channel, spatial_size, hist_bins,
               spatial_feat, hist_feat, hog_feat, window_side, scaler):
    """
    Save the trained model with parameters and scaler. 
    """
    data = {'color_space': color_space,
            'orient': orient,
            'pix_per_cell': pix_per_cell,
            'cell_per_block': cell_per_block,
            'hog_channel': hog_channel,
            'spatial_size': spatial_size,
            'hist_bins': hist_bins,
            'spatial_feat': spatial_feat,
            'hist_feat': hist_feat,
            'hog_feat': hog_feat,
            'window_side': window_side,
            'clf': clf,
            'scaler': scaler}
    pickle.dump(data, open(data_file, 'wb'))

if __name__ == '__main__':
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: train_svm dataset_dir [output_filename]")
        sys.exit(0)

    # Extract features and generate labels
    dataset_dir = sys.argv[1]
    cars_filename = glob.glob(os.path.join(dataset_dir, 'cars', '*'))
    noncars_filename = glob.glob(os.path.join(dataset_dir, 'noncars', '*'))
    tu.balance_lists(cars_filename, noncars_filename)

    print("Extracting features...")
    start = time.time()
    cars = extract_features(cars_filename)
    noncars = extract_features(noncars_filename, (window_side, window_side))
    time_elapsed = time.time() - start
    print("Feature extraction time: %f" % time_elapsed)
    features = np.vstack((noncars, cars))
    labels = tu.gen_binary_labels((len(noncars_filename), len(cars_filename)))

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size)

    # Scale features
    X_scaler = StandardScaler().fit(X_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    print("Training data size: %d" % len(X_train))
    print("Test data size: %d" % len(X_test))

    # Train the model
    start = time.time()
    clf = train_model_cv(X_train_scaled, y_train, Cs)
    time_elapsed = time.time() - start
    print("Training time: %f" % time_elapsed)
    print("Best parameters: %s" % str(clf.best_params_))
    print(" Train accuracy: %f" % clf.score(X_train_scaled, y_train))
    print("  Test accuracy: %f" % clf.score(X_test_scaled, y_test))

    # Save the model
    if len(sys.argv) >= 3:
        save_model(sys.argv[2], clf, color_space, orient, pix_per_cell,
                   cell_per_block, hog_channel, spatial_size, hist_bins,
                   spatial_feat, hist_feat, hog_feat, window_side, X_scaler)

