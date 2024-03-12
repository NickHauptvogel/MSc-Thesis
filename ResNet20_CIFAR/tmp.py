from __future__ import print_function

from keras.optimizers import Adam, SGD
from keras.datasets import cifar10
import numpy as np
import argparse
import os
import json
import pickle

from ResNet20 import resnet_v1, resnet_v2

# Add parent directory to path
import sys
sys.path.append('..')

from dataset_split import split_dataset

def main(path):

    # Training parameters
    num_classes = 10

    # Subtracting pixel mean improves accuracy
    subtract_pixel_mean = True

    # Model parameter
    # ----------------------------------------------------------------------------
    #           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
    # Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
    #           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
    # ----------------------------------------------------------------------------
    # ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
    # ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
    # ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
    # ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
    # ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
    # ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
    # ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
    # ---------------------------------------------------------------------------
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
    (x_train, _), _ = cifar10.load_data()

    # Input image dimensions.
    input_shape = x_train.shape[1:]

    # Normalize data.
    x_train = x_train.astype('float32') / 255

    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean

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

    # Load best model weights (already saved)
    model_path = [f.path for f in os.scandir(path) if f.name.endswith('.h5')][0]
    model.load_weights(model_path)

    # Save predictions
    y_pred = model.predict(x_val)
    test_predictions_name = [f.name for f in os.scandir(path) if f.name.endswith('test_predictions.pkl')][0]
    val_predictions_name = test_predictions_name.replace('test', 'val')
    with open(val_predictions_name, 'wb') as f:
        pickle.dump(y_pred, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a ResNet model on CIFAR10')
    parser.add_argument('--path', type=str, default='results/50_independent_wenzel_no_checkp_bootstr', help='Folder with the models')
    args = parser.parse_args()

    # Get all subdirectories in path
    subdirs = [f.path for f in os.scandir(args.path) if f.is_dir()]
    for subdir in subdirs:
        main(subdir)