import os
import cv2
import sys
import glob
import pickle
import numpy as np
import utils as tu
import feature_extraction as fx

from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


color_space = 'YCrCb'  # Can be GRAY, RGB, HSV, LUV, HLS, YUV, YCrCb
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

loss = 'binary_crossentropy'
optimizer = 'adam'
test_size = 0.2
batch_size = 128
epochs = 10


def extract_feature(image, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to contain different feature for later concatenation
    sub_features = []
    feature_image = fx.convert_image(image, color_space)

    if spatial_feat:
        spatial_features = fx.bin_spatial(feature_image, size=spatial_size)
        sub_features.append(spatial_features)

    if hist_feat:
        hist_features = fx.color_hist(feature_image, nbins=hist_bins)
        sub_features.append(hist_features)

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
    features = []
    global color_space, orient, pix_per_cell, cell_per_block, hog_channel
    global spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat, window_side
    for name in image_filenames:
        image = cv2.imread(name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if resize:
            image = cv2.resize(image, resize)
        features.append(extract_feature(image, color_space, spatial_size, hist_bins, orient,
                                        pix_per_cell, cell_per_block, hog_channel,
                                        spatial_feat, hist_feat, hog_feat))
    return np.array(features)


def define_model(input_dim):
    model = Sequential()
    model.add(Dense(200, activation='tanh', input_dim=input_dim))
    model.add(Dropout(p=0.5))
    model.add(Dense(50, activation='tanh'))
    model.add(Dropout(p=0.5))
    model.add(Dense(10, activation='tanh'))
    model.add(Dropout(p=0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model


def get_data_generator(data_pairs, scaler, batch_size=128):
    """
    Creates a batch data generator.

    :param data_pairs: list of training image filenames
    :return: a generator which yields mini-batches of shuffled (X, y) pairs.
    """
    num_data = len(data_pairs)
    while 1:  # Loop forever so the generator never terminates
        # shuffle the image filenames at each epoch
        shuffle(data_pairs)
        for offset in range(0, num_data, batch_size):
            batch_pairs = data_pairs[offset:offset + batch_size]

            #images = [cv2.cvtColor(cv2.imread(dp[0]), cv2.COLOR_BGR2RGB) for dp in batch_pairs]
            labels = [dp[1] for dp in batch_pairs]

            #X = np.array(images)
            X = extract_features([dp[0] for dp in batch_pairs])
            y = np.array(labels)
            yield shuffle(scaler.transform(X), y)


def save_model(data_file, model_file, color_space, orient,
               pix_per_cell, cell_per_block,
               hog_channel, spatial_size, hist_bins,
               spatial_feat, hist_feat, hog_feat, window_side, scaler):
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
            'model': model_file,
            'scaler': scaler}
    pickle.dump(data, open(data_file, 'wb'))


def get_scaler(train_data_pairs):
    image_names = [dp[0] for dp in train_data_pairs]
    features = extract_features(image_names)
    scaler = StandardScaler().fit(features)
    return scaler


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: train_ann dataset_dir model_output")

    dataset_dir = sys.argv[1]
    model_file = sys.argv[2]
    car_names = glob.glob(os.path.join(dataset_dir, 'cars', '*'))
    noncar_names = glob.glob(os.path.join(dataset_dir, 'noncars', '*'))
    tu.balance_lists(noncar_names, car_names)
    labels = tu.gen_binary_labels((len(noncar_names), len(car_names)))
    names = noncar_names + car_names
    data_pairs = list(zip(names, labels))

    train_data_pairs, valid_data_pairs = train_test_split(data_pairs,
                                                          test_size=test_size)
    scaler = get_scaler(train_data_pairs)
    train_gen = get_data_generator(train_data_pairs, scaler)
    valid_gen = get_data_generator(valid_data_pairs, scaler)

    input_dim = len(extract_feature(cv2.cvtColor(cv2.imread(train_data_pairs[0][0]),
                                                 cv2.COLOR_BGR2RGB),
                                    color_space, spatial_size, hist_bins,
                                    orient, pix_per_cell, cell_per_block, hog_channel,
                                    spatial_feat, hist_feat, hog_feat))
    # Define model
    model = define_model(input_dim)
    # Compile model
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    # Train model
    model.fit_generator(generator=train_gen,
                        validation_data=valid_gen,
                        nb_epoch=epochs,
                        samples_per_epoch=len(train_data_pairs),
                        nb_val_samples=len(valid_data_pairs))
    model.save(model_file)
    save_model('model.ann', model_file, color_space, orient,
               pix_per_cell, cell_per_block,
               hog_channel, spatial_size, hist_bins,
               spatial_feat, hist_feat, hog_feat, window_side, scaler)
