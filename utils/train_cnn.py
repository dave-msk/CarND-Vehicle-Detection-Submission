import os
import sys
import glob
import cv2
import pickle
import numpy as np
import utils as tu

from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda
from keras.layers import Conv2D, Dropout, MaxPooling2D

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

input_size = (64, 64, 3)

loss = 'binary_crossentropy'
optimizer = 'adam'
test_size = 0.2
batch_size = 128
epochs = 10


def define_model():
    """
    Defines the architecture of network.
    :return: CNN model for training
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1, input_shape=input_size))
    model.add(Conv2D(16, 5, 5, activation='relu'))
    model.add(Dropout(p=0.2))
    model.add(Conv2D(32, 5, 5, activation='relu'))
    model.add(Dropout(p=0.3))
    model.add(Conv2D(64, 5, 5, activation='relu'))
    model.add(Flatten())
    model.add(Dropout(p=0.4))
    model.add(Dense(200, activation='tanh'))
    model.add(Dropout(p=0.5))
    model.add(Dense(50, activation='tanh'))
    model.add(Dropout(p=0.5))
    model.add(Dense(10, activation='tanh'))
    model.add(Dropout(p=0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model


def get_data_generator(data_pairs, batch_size=128):
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
            batch_pairs = data_pairs[offset:offset+batch_size]

            images = [cv2.cvtColor(cv2.imread(dp[0]), cv2.COLOR_BGR2RGB) for dp in batch_pairs]
            labels = [dp[1] for dp in batch_pairs]

            X = np.array(images)
            y = np.array(labels)
            yield shuffle(X, y)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: train_cnn dataset_dir model_output")
        sys.exit(0)

    dataset_dir = sys.argv[1]
    cars_wildcard = os.path.join(dataset_dir, 'cars', '*')
    car_names = glob.glob(cars_wildcard)
    noncars_wildcard = os.path.join(dataset_dir, 'noncars', '*')
    noncar_names = glob.glob(noncars_wildcard)
    tu.balance_lists(car_names, noncar_names)
    labels = tu.gen_binary_labels((len(noncar_names), len(car_names)))
    names = noncar_names + car_names
    data_pairs = list(zip(names, labels))

    train_data_pairs, valid_data_pairs = train_test_split(data_pairs,
                                                          test_size=test_size)
    nb_train_data = len(train_data_pairs)
    nb_valid_data = len(valid_data_pairs)

    train_data = get_data_generator(train_data_pairs, batch_size=batch_size)
    valid_data = get_data_generator(valid_data_pairs, batch_size=batch_size)

    # create model
    model = define_model()
    # compile model
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    # train model
    model.fit_generator(generator=train_data,
                        validation_data=valid_data,
                        samples_per_epoch=nb_train_data,
                        nb_epoch=epochs,
                        nb_val_samples=nb_valid_data)
    # save model
    model_file = sys.argv[2]
    model.save(model_file)

    data = {'model': model_file, 'input_size': input_size}
    pickle.dump(data, open('model.cnn', 'wb'))
