from __future__ import print_function

import tensorflow as tf
import keras
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import argparse
import os
from tqdm.keras import TqdmCallback
from datetime import datetime
import json
import pickle

from ResNet20 import resnet_v1, resnet_v2

# Add parent directory to path
import sys
sys.path.append('..')

from load_data import load_cifar
from dataset_split import split_dataset
from snapshot_lr_schedule import sse_lr_schedule

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, default='01', help='ID of the experiment')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--out_folder', type=str, default='results', help='output folder')
# 32 in other implementations
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--validation_split', type=float, default=0.1, help='validation split')
parser.add_argument('--checkpointing', action='store_true', help='save the best model during training')
parser.add_argument('--checkpoint_every', type=int, default=-1, help='save the model every x epochs')
parser.add_argument('--hold_out_validation_split', type=float, default=0.0, help='fraction of validation set to hold out for final eval')
parser.add_argument('--data_augmentation', action='store_true', help='use data augmentation')
# 0.1 in other implementations
parser.add_argument('--augm_shift', type=float, default=4, help='augmentation shift (px for >1 or fraction <1)')
# 1e-3 in other implementations
parser.add_argument('--initial_lr', type=float, default=0.1, help='initial learning rate')
# 1e-4 in other implementations
parser.add_argument('--l2_reg', type=float, default=0.002, help='l2 regularization')
# Adam in other implementations
parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
parser.add_argument('--nesterov', action='store_true', help='use Nesterov momentum')
parser.add_argument('--bootstrapping', action='store_true', help='use bootstrapping')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes for CIFAR (10/100)')
parser.add_argument('--SSE_lr', action='store_true', help='learning rate for SSE. Use with checkpoint_every for M, initial_lr for reset learning rate and number of epochs for B')
parser.add_argument('--test_time_augmentation', action='store_true', help='use test time augmentation')
parser.add_argument('--debug', action='store_true', help='debug mode')

args = parser.parse_args()

# Random seed
seed = args.seed
tf.random.set_seed(seed)
np.random.seed(seed)

out_folder = args.out_folder
os.makedirs(out_folder, exist_ok=True)
experiment_id = args.id
batch_size = args.batch_size
epochs = args.epochs
validation_split = args.validation_split
hold_out_validation_split = args.hold_out_validation_split
checkpointing = args.checkpointing
checkpoint_every = args.checkpoint_every
data_augmentation = args.data_augmentation
augm_shift = args.augm_shift
initial_lr = args.initial_lr
l2_reg = args.l2_reg
optimizer = args.optimizer
momentum = args.momentum
nesterov = args.nesterov
bootstrapping = args.bootstrapping
num_classes = args.num_classes
SSE_lr = args.SSE_lr
test_time_augmentation = args.test_time_augmentation
debug = args.debug

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

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)

# Prepare model saving directory.
current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = os.path.join(os.getcwd(), out_folder, current_date + f'_{experiment_id}')
model_name = f'{experiment_id}_cifar{str(num_classes)}_{model_type}'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name + '.h5')

# Configuration
configuration = args.__dict__.copy()
configuration['model'] = model_type
configuration['tf_version'] = tf.__version__
configuration['keras_version'] = keras.__version__
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
  details = tf.config.experimental.get_device_details(gpu_devices[0])
  configuration['GPU'] = details.get('device_name', 'Unknown GPU')
print(configuration)

# Load the CIFAR data.
(x_train, y_train), (x_test, y_test) = load_cifar(num_classes=num_classes, subtract_pixel_mean=subtract_pixel_mean, debug=debug)

# Split the training data into a training and a validation set
if validation_split > 0 or bootstrapping:
    train_indices, val_indices = split_dataset(x_train.shape[0], validation_split, bootstrap=bootstrapping, random=True)
    if hold_out_validation_split > 0.0:
        holdout_size = int(len(val_indices) * hold_out_validation_split)
        holdout_indices = val_indices[:holdout_size]
        val_indices = val_indices[holdout_size:]
        x_val_holdout, y_val_holdout = x_train[holdout_indices], y_train[holdout_indices]
        configuration['holdout_indices'] = holdout_indices.tolist()
        # Make sure the three sets are disjoint
        assert len(np.intersect1d(train_indices, holdout_indices)) == 0
        assert len(np.intersect1d(val_indices, holdout_indices)) == 0
    assert len(np.intersect1d(train_indices, val_indices)) == 0
    x_val, y_val = x_train[val_indices], y_train[val_indices]
    x_train, y_train = x_train[train_indices], y_train[train_indices]
    configuration['train_indices'] = train_indices.tolist()
    configuration['val_indices'] = val_indices.tolist()
else:
    x_val, y_val = x_test, y_test
    print('Using test set as validation set')
    if checkpointing:
        print("WARNING! YOU ARE VALIDATING ON THE TEST SET AND CHECKPOINTING IS ENABLED! SELECTION BIAS")
        sys.exit(1)

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_val shape:', x_val.shape)
print('y_val shape:', y_val.shape)
if hold_out_validation_split > 0:
    print('x_val_holdout shape:', x_val_holdout.shape)
    print('y_val_holdout shape:', y_val_holdout.shape)

# Save configuration to json
fn = os.path.join(save_dir, model_name + '_config.json')
with open(fn, 'w') as f:
    json.dump(configuration, f, indent=4)

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs. Expressed relatively to 200 epochs as
    8/20, 12/20, 16/20, 18/20.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = initial_lr
    if epoch > int(0.9 * epochs):
        lr *= 0.5e-3
    elif epoch > int(0.8 * epochs):
        lr *= 1e-3
    elif epoch > int(0.6 * epochs):
        lr *= 1e-2
    elif epoch > int(0.4 * epochs):
        lr *= 1e-1
    return lr

# Input image dimensions.
input_shape = x_train.shape[1:]

if version == 2:
    model = resnet_v2(input_shape=input_shape, depth=depth, l2_reg=l2_reg, num_classes=num_classes)
else:
    model = resnet_v1(input_shape=input_shape, depth=depth, l2_reg=l2_reg, num_classes=num_classes)

if optimizer == 'adam':
    optimizer_ = Adam(learning_rate=initial_lr)
elif optimizer == 'sgd':
    optimizer_ = SGD(learning_rate=initial_lr, momentum=momentum, nesterov=nesterov)
else:
    raise ValueError('Unknown optimizer')

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer_,
              metrics=['accuracy'])
#model.summary()
print(model_type)

# Prepare callbacks
callbacks = []
if checkpointing:
    if checkpoint_every > 0:
        filepath_preformat = os.path.join(save_dir, model_name + '_{epoch:03d}.h5')
        checkpoint = ModelCheckpoint(filepath=filepath_preformat,
                                     monitor='val_accuracy',
                                     save_weights_only=True,
                                     save_freq='epoch',
                                     save_best_only=False)
    else:
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     monitor='val_accuracy',
                                     save_weights_only=True,
                                     save_best_only=True)
    callbacks.append(checkpoint)

if SSE_lr:
    if checkpoint_every < 1:
        print('ERROR: checkpoint_every must be set to a positive integer when using SSE_lr')
        sys.exit(1)
    M = epochs // checkpoint_every
    lr_scheduler = LearningRateScheduler(lambda epoch: sse_lr_schedule(epoch, B=epochs, M=M, initial_lr=initial_lr))
else:
    lr_scheduler = LearningRateScheduler(lr_schedule)

#lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
#                               cooldown=0,
#                               patience=5,
#                               min_lr=0.5e-6)

tqdm_callback = TqdmCallback(verbose=0)

callbacks.extend([lr_scheduler, tqdm_callback])

datagen = ImageDataGenerator(
    # randomly shift images horizontally
    width_shift_range=augm_shift,
    # randomly shift images vertically
    height_shift_range=augm_shift,
    # randomly flip images
    horizontal_flip=True)

if not data_augmentation:
    print('Not using data augmentation.')
    # Run training, with or without data augmentation.
    history = model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        shuffle=True,
        verbose=0,
        callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        validation_data=(x_val, y_val),
        epochs=epochs,
        verbose=0,
        workers=4,
        callbacks=callbacks)

if not checkpointing:
    # Save the model
    model.save(filepath)

# Get all model checkpoint files
checkpoint_files = sorted([f for f in os.listdir(save_dir) if f.startswith(model_name) and f.endswith('.h5')])
if checkpoint_every > 0:
    # Clean up the checkpoint files to include every x epochs
    for  i, file in enumerate(checkpoint_files):
        if (i+1) % checkpoint_every != 0:
            os.remove(os.path.join(save_dir, file))

checkpoint_files = sorted([f for f in os.listdir(save_dir) if f.startswith(model_name) and f.endswith('.h5')])

if len(checkpoint_files) == 1:
    print('Only one model saved')

scores = {'history': history.history, 'test_loss': [], 'test_accuracy': [], 'val_loss': [], 'val_accuracy': []}
if hold_out_validation_split > 0:
    scores['holdout_loss'] = []
    scores['holdout_accuracy'] = []

for file in checkpoint_files:
    print('\nLoading model:', file)
    file_model_name = file.replace('.h5', '')
    # Load the model
    model.load_weights(os.path.join(save_dir, file))
    # Score trained model.
    if test_time_augmentation:
        score, acc = model.evaluate(datagen.flow(x_test, y_test), verbose=0)
        y_pred = model.predict(datagen.flow(x_test))
    else:
        score, acc = model.evaluate(x_test, y_test, verbose=0)
        y_pred = model.predict(x_test)
    print('Test score:', score)
    print('Test accuracy:', acc)
    scores['test_loss'].append(score)
    scores['test_accuracy'].append(acc)
    # Save predictions
    fn = os.path.join(save_dir, file_model_name + '_test_predictions.pkl')
    with open(fn, 'wb') as f:
        pickle.dump(y_pred, f)

    if validation_split > 0 or bootstrapping:
        if test_time_augmentation:
            val_score, val_acc = model.evaluate(datagen.flow(x_val, y_val), verbose=0)
            y_pred = model.predict(datagen.flow(x_val))
        else:
            val_score, val_acc = model.evaluate(x_val, y_val, verbose=0)
            y_pred = model.predict(x_val)
        print('Val score:', val_score)
        print('Val accuracy:', val_acc)
        scores['val_loss'].append(val_score)
        scores['val_accuracy'].append(val_acc)
        fn = os.path.join(save_dir, file_model_name + '_val_predictions.pkl')
        with open(fn, 'wb') as f:
            pickle.dump(y_pred, f)

        if hold_out_validation_split > 0:
            if test_time_augmentation:
                holdout_score, holdout_acc = model.evaluate(datagen.flow(x_val_holdout, y_val_holdout), verbose=0)
                y_pred = model.predict(datagen.flow(x_val_holdout))
            else:
                holdout_score, holdout_acc = model.evaluate(x_val_holdout, y_val_holdout, verbose=0)
                y_pred = model.predict(x_val_holdout)
            print('Holdout score:', holdout_score)
            print('Holdout accuracy:', holdout_acc)
            scores['holdout_loss'].append(holdout_score)
            scores['holdout_accuracy'].append(holdout_acc)
            fn = os.path.join(save_dir, file_model_name + '_val_holdout_predictions.pkl')
            with open(fn, 'wb') as f:
                pickle.dump(y_pred, f)

# Save as dictionary
fn = os.path.join(save_dir, model_name + '_scores.json')
# Change all np.float32 to float
for k, v in scores['history'].items():
    scores['history'][k] = [float(x) for x in v]
with open(fn, 'w') as f:
    json.dump(scores, f, indent=4)
