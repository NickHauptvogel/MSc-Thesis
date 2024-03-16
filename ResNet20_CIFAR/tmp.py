from __future__ import print_function

from keras.optimizers import SGD
import numpy as np
import argparse
import os
import json
import pickle
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator

from ResNet import resnet_v1, resnet_v2

# Add parent directory to path
import sys
sys.path.append('..')

from load_data import load_cifar

def posttrain_tta_predictions(path):

    num_classes = 10
    subtract_pixel_mean = True
    n = 3

    # Model version
    # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
    version = 1

    # Computed depth from supplied model parameter n
    if version == 1:
        depth = n * 6 + 2
    elif version == 2:
        depth = n * 9 + 2

    # Load config file (Search for file that ends with config.json)
    config_file = [f.path for f in os.scandir(path) if f.name.endswith('config.json')][0]
    with open(config_file, 'r') as f:
        configuration = json.load(f)

    # Load the CIFAR10 data.
    _, (x_test, _) = load_cifar(num_classes, subtract_pixel_mean, False)

    # Input image dimensions.
    input_shape = x_test.shape[1:]

    l2_reg = configuration['l2_reg']
    augm_shift = configuration['augm_shift']

    if version == 2:
        model = resnet_v2(input_shape=input_shape, l2_reg=l2_reg, depth=depth)
    else:
        model = resnet_v1(input_shape=input_shape, l2_reg=l2_reg, depth=depth)

    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(learning_rate=0),
                  metrics=['accuracy'])

    # Load best model weights
    model_path = sorted([f.path for f in os.scandir(path) if f.name.endswith('.h5')])
    if len(model_path) > 1:
        print('Multiple model files found, using the last one')
        model_path = model_path[-1]
    else:
        model_path = model_path[0]

    model.load_weights(model_path)

    datagen = ImageDataGenerator(
        # randomly shift images horizontally
        width_shift_range=augm_shift,
        # randomly shift images vertically
        height_shift_range=augm_shift,
        # randomly flip images
        horizontal_flip=True)

    y_pred = model.predict(datagen.flow(x_test, shuffle=False))

    test_predictions_name = [f.name for f in os.scandir(path) if f.name.endswith('test_predictions.pkl')][0]
    tta_predictions_name = test_predictions_name.replace('test', 'test_tta')
    with open(os.path.join(path, tta_predictions_name), 'wb') as f:
        pickle.dump(y_pred, f)


def posttrain_val_predictions(path):

    num_classes = 10
    subtract_pixel_mean = True
    n = 3

    # Model version
    # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
    version = 1

    # Computed depth from supplied model parameter n
    if version == 1:
        depth = n * 6 + 2
    elif version == 2:
        depth = n * 9 + 2

    # Load config file (Search for file that ends with config.json)
    config_file = [f.path for f in os.scandir(path) if f.name.endswith('config.json')][0]
    with open(config_file, 'r') as f:
        configuration = json.load(f)

    # Load the CIFAR10 data.
    (x_train, _), _ = load_cifar(num_classes, subtract_pixel_mean, False)

    # Input image dimensions.
    input_shape = x_train.shape[1:]

    # Split the training data into a training and a validation set
    if configuration.get('val_indices') is None:
        print('No validation indices found in configuration file')
        sys.exit(1)

    val_indices = configuration['val_indices']
    x_val = x_train[val_indices]
    train_indices = configuration['train_indices']
    x_train = x_train[train_indices]

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'validation samples')

    l2_reg = configuration['l2_reg']

    if version == 2:
        model = resnet_v2(input_shape=input_shape, l2_reg=l2_reg, depth=depth)
    else:
        model = resnet_v1(input_shape=input_shape, l2_reg=l2_reg, depth=depth)

    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(learning_rate=0),
                  metrics=['accuracy'])

    # Load best model weights
    model_path = sorted([f.path for f in os.scandir(path) if f.name.endswith('.h5')])[0]
    model.load_weights(model_path)

    # Save predictions
    y_pred = model.predict(x_val)
    test_predictions_name = [f.name for f in os.scandir(path) if f.name.endswith('test_predictions.pkl')][0]
    val_predictions_name = test_predictions_name.replace('test', 'val')
    with open(val_predictions_name, 'wb') as f:
        pickle.dump(y_pred, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='results/50_independent_wenzel_no_checkp_no_val', help='Folder with the models')
    args = parser.parse_args()

    # Get all subdirectories in path
    subdirs = sorted([f.path for f in os.scandir(args.path) if f.is_dir()])
    for subdir in tqdm(subdirs):
        #posttrain_val_predictions(subdir)
        posttrain_tta_predictions(subdir)
