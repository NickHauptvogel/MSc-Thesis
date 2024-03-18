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

from ResNet import resnet_v1, resnet_v2

# Add parent directory to path
import sys
sys.path.append('..')

from load_data import load_cifar
from dataset_split import split_dataset
from lr_schedules import cifar_schedule, sse_lr_schedule, step_decay_schedule

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
parser.add_argument('--model_type', type=str, default='ResNet20v1', help='model type')
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
parser.add_argument('--use_case', type=str, default='cifar10', help='Use case. Supported: cifar10, cifar100, retinopathy')
parser.add_argument('--lr_schedule', type=str, default='cifar', help='Learning rate schedule. Supported: cifar, sse, retinopathy')
parser.add_argument('--test_time_augmentation', action='store_true', help='use test time augmentation')
parser.add_argument('--store_models', action='store_true', help='store all models')
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
model_type = args.model_type
data_augmentation = args.data_augmentation
augm_shift = args.augm_shift
initial_lr = args.initial_lr
l2_reg = args.l2_reg
optimizer = args.optimizer
momentum = args.momentum
nesterov = args.nesterov
bootstrapping = args.bootstrapping
use_case = args.use_case
lr_schedule = args.lr_schedule
test_time_augmentation = args.test_time_augmentation
store_models = args.store_models
debug = args.debug

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

if model_type == 'ResNet20v1':
    depth = 20
    version = 1
elif model_type == 'ResNet110v1':
    depth = 110
    version = 1
elif model_type == 'ResNet50v1':
    depth = 50
    version = 1
else:
    raise ValueError('Unknown model type: ' + model_type)

# Prepare model saving directory.
current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = os.path.join(os.getcwd(), out_folder, current_date + f'_{experiment_id}')
model_name = f'{experiment_id}_{use_case}_{model_type}'
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

if use_case == 'cifar10' or use_case == 'cifar100':
    if use_case == 'cifar10':
        num_classes = 10
    else:
        num_classes = 100
    # Load the CIFAR data.
    (x_train, y_train), (x_test, y_test) = load_cifar(num_classes=num_classes, subtract_pixel_mean=True, debug=debug)
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

    if data_augmentation:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            # randomly shift images horizontally
            width_shift_range=augm_shift,
            # randomly shift images vertically
            height_shift_range=augm_shift,
            # randomly flip images
            horizontal_flip=True)
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)
    train_loader = train_datagen.flow(x_train, y_train, batch_size=batch_size)

    val_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=hold_out_validation_split)
    val_loader = val_datagen.flow(x_val, y_val, batch_size=batch_size, shuffle=False, subset='training')
    val_loader_x_only = val_datagen.flow(x_val, batch_size=batch_size, shuffle=False, subset='training')
    if hold_out_validation_split > 0:
        val_holdout_loader = val_datagen.flow(x_val_holdout, y_val_holdout, batch_size=batch_size, shuffle=False, subset='validation')
        val_holdout_loader_x_only = val_datagen.flow(x_val_holdout, batch_size=batch_size, shuffle=False, subset='validation')

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    if test_time_augmentation:
        test_tta_data = ImageDataGenerator(
            rescale=1. / 255,
            # randomly shift images horizontally
            width_shift_range=augm_shift,
            # randomly shift images vertically
            height_shift_range=augm_shift,
            # randomly flip images
            horizontal_flip=True)
        test_tta_loader = test_tta_data.flow(x_test, y_test, batch_size=batch_size, shuffle=False)
        test_tta_loader_x_only = test_tta_data.flow(x_test, batch_size=batch_size, shuffle=False)
    test_loader = test_datagen.flow(x_test, y_test, batch_size=batch_size, shuffle=False)
    test_loader_x_only = test_datagen.flow(x_test, batch_size=batch_size, shuffle=False)

elif use_case == 'retinopathy':
    dataset_path = '../../Datasets/Diabetic_Retinopathy'
    # Load the retinopathy data.
    num_classes = 1
    target_size = (256, 256)
    if data_augmentation:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            # randomly shift images horizontally
            width_shift_range=augm_shift,
            # randomly shift images vertically
            height_shift_range=augm_shift,
            # randomly flip images
            horizontal_flip=True)
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)
    train_loader = train_datagen.flow_from_directory(f'{dataset_path}/train', target_size=target_size, batch_size=batch_size, subset='training', class_mode='binary')

    val_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=hold_out_validation_split)
    val_loader = val_datagen.flow_from_directory(f'{dataset_path}/validation', target_size=target_size, batch_size=batch_size, subset='training', shuffle=False, class_mode='binary')
    val_loader_x_only = val_datagen.flow_from_directory(f'{dataset_path}/validation', target_size=target_size, batch_size=batch_size, subset='training', shuffle=False, class_mode=None)
    if hold_out_validation_split > 0:
        val_holdout_loader = val_datagen.flow_from_directory(f'{dataset_path}/validation', target_size=target_size, batch_size=batch_size, subset='validation', shuffle=False, class_mode='binary')
        val_holdout_loader_x_only = val_datagen.flow_from_directory(f'{dataset_path}/validation', target_size=target_size, batch_size=batch_size, subset='validation', shuffle=False, class_mode=None)

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    if test_time_augmentation:
        test_tta_data = ImageDataGenerator(
            rescale=1. / 255,
            # randomly shift images horizontally
            width_shift_range=augm_shift,
            # randomly shift images vertically
            height_shift_range=augm_shift,
            # randomly flip images
            horizontal_flip=True)
        test_tta_loader = test_tta_data.flow_from_directory(f'{dataset_path}/test', target_size=target_size, batch_size=batch_size, shuffle=False, class_mode='binary')
        test_tta_loader_x_only = test_tta_data.flow_from_directory(f'{dataset_path}/test', target_size=target_size, batch_size=batch_size, shuffle=False, class_mode=None)
    test_loader = test_datagen.flow_from_directory(f'{dataset_path}/test', target_size=target_size, batch_size=batch_size, shuffle=False, class_mode='binary')
    test_loader_x_only = test_datagen.flow_from_directory(f'{dataset_path}/test', target_size=target_size, batch_size=batch_size, shuffle=False, class_mode=None)

    # Set validation split such that it is used for predictions
    validation_split = val_loader.samples / (val_loader.samples + train_loader.samples)
else:
    raise ValueError('Unknown use case: ' + use_case)


print('x_train samples:', train_loader.samples)
print('x_val samples:', val_loader.samples)
if hold_out_validation_split > 0:
    print('x_val_holdout samples:', val_holdout_loader.samples)
print('x_test samples:', test_loader.samples)

x_train, y_train = next(train_loader)
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)

# Save configuration to json
fn = os.path.join(save_dir, model_name + '_config.json')
with open(fn, 'w') as f:
    json.dump(configuration, f, indent=4)

# Input image dimensions from train loader
input_shape = train_loader.image_shape

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

if num_classes == 1:
    loss_ = 'binary_crossentropy'
else:
    loss_ = 'categorical_crossentropy'

model.compile(loss=loss_,
              optimizer=optimizer_,
              metrics=['accuracy'])

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

if lr_schedule == 'sse':
    if checkpoint_every < 1:
        print('ERROR: checkpoint_every must be set to a positive integer when using sse lr schedule')
        sys.exit(1)
    M = epochs // checkpoint_every
    lr_scheduler = LearningRateScheduler(lambda epoch: sse_lr_schedule(epoch, B=epochs, M=M, initial_lr=initial_lr))
elif lr_schedule == 'cifar':
    lr_scheduler = LearningRateScheduler(lambda epoch: cifar_schedule(epoch, initial_lr, epochs))
elif lr_schedule == 'retinopathy':
    # For now: Use values from the paper
    decay_epochs = [
        (int(start_epoch_str) * epochs) // 90
        for start_epoch_str in ['30', '60']
    ]
    decay_ratio = 0.2
    warmup_epochs = 1
    lr_scheduler = LearningRateScheduler(lambda epoch: step_decay_schedule(epoch, initial_lr, decay_ratio, decay_epochs, warmup_epochs))

#lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
#                               cooldown=0,
#                               patience=5,
#                               min_lr=0.5e-6)

tqdm_callback = TqdmCallback(verbose=0)

callbacks.extend([lr_scheduler, tqdm_callback])

history = model.fit(
    train_loader,
    validation_data=val_loader,
    epochs=epochs,
    verbose=0,
    callbacks=callbacks)

if not checkpointing:
    # Save the model in the end
    model.save(filepath)

# Get all model checkpoint files
checkpoint_files = sorted([f for f in os.listdir(save_dir) if f.startswith(model_name) and f.endswith('.h5')])
if checkpoint_every > 0:
    # Clean up the checkpoint files to include every x epochs
    for i, file in enumerate(checkpoint_files):
        if (i+1) % checkpoint_every != 0:
            os.remove(os.path.join(save_dir, file))

checkpoint_files = sorted([f for f in os.listdir(save_dir) if f.startswith(model_name) and f.endswith('.h5')])

if len(checkpoint_files) == 1:
    print('Only one model saved')

scores = {'history': history.history, 'test_loss': [], 'test_accuracy': [], 'val_loss': [], 'val_accuracy': []}
if hold_out_validation_split > 0:
    scores['holdout_loss'] = []
    scores['holdout_accuracy'] = []
if test_time_augmentation:
    scores['tta_test_loss'] = []
    scores['tta_test_accuracy'] = []

for file in checkpoint_files:
    print('\nLoading model:', file)
    file_model_name = file.replace('.h5', '')
    # Load the model
    model.load_weights(os.path.join(save_dir, file))
    # Score trained model.
    score, acc = model.evaluate(test_loader, verbose=0)
    y_pred = model.predict(test_loader_x_only)
    print('Test score:', score)
    print('Test accuracy:', acc)
    scores['test_loss'].append(score)
    scores['test_accuracy'].append(acc)
    # Save predictions
    fn = os.path.join(save_dir, file_model_name + '_test_predictions.pkl')
    with open(fn, 'wb') as f:
        pickle.dump(y_pred, f)

    if test_time_augmentation:
        score, acc = model.evaluate(test_tta_loader, verbose=0)
        y_pred = model.predict(test_tta_loader_x_only)
        print('TTA Test score:', score)
        print('TTA Test accuracy:', acc)
        scores['test_tta_loss'].append(score)
        scores['test_tta_accuracy'].append(acc)
        # Save predictions
        fn = os.path.join(save_dir, file_model_name + '_test_tta_predictions.pkl')
        with open(fn, 'wb') as f:
            pickle.dump(y_pred, f)

    if validation_split > 0 or bootstrapping:
        val_score, val_acc = model.evaluate(val_loader, verbose=0)
        y_pred = model.predict(val_loader_x_only)
        print('Val score:', val_score)
        print('Val accuracy:', val_acc)
        scores['val_loss'].append(val_score)
        scores['val_accuracy'].append(val_acc)
        fn = os.path.join(save_dir, file_model_name + '_val_predictions.pkl')
        with open(fn, 'wb') as f:
            pickle.dump(y_pred, f)

        if hold_out_validation_split > 0:
            holdout_score, holdout_acc = model.evaluate(val_holdout_loader, verbose=0)
            y_pred = model.predict(val_holdout_loader_x_only)
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

if not store_models:
    # Clean up the checkpoint files
    for file in checkpoint_files:
        os.remove(os.path.join(save_dir, file))
