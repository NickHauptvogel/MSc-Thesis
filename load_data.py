import keras
from keras.datasets import cifar10, cifar100
import numpy as np

from dataset_split import split_dataset

def load_cifar(num_classes, subtract_pixel_mean, debug):
    # Load the CIFAR10 data.
    if num_classes == 10:
        dataloader = cifar10
    elif num_classes == 100:
        dataloader = cifar100
    else:
        raise ValueError('Unknown number of classes')
    (x_train, y_train), (x_test, y_test) = dataloader.load_data()

    if debug:
        x_train = x_train[:1000]
        y_train = y_train[:1000]
        x_test = x_test[:100]
        y_test = y_test[:100]

    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)

