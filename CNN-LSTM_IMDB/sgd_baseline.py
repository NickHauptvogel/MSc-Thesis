'''Train a recurrent convolutional network on the IMDB sentiment
classification task.

Gets to 0.8498 test accuracy after 2 epochs. 41s/epoch on K520 GPU.
'''
from __future__ import print_function

import tensorflow as tf
import keras
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import SGD
from keras.datasets import imdb
import numpy as np
import argparse
import os
from tqdm.keras import TqdmCallback
from datetime import datetime
import json
import pickle

import priorfactory
# Add parent directory to path
import sys
sys.path.append('..')

from dataset_split import split_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, default='01', help='ID of the experiment')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--out_folder', type=str, default='results', help='output folder')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--validation_split', type=float, default=0.2, help='validation split')
parser.add_argument('--checkpointing', action='store_true', help='save the best model during training')
parser.add_argument('--hold_out_validation_split', type=float, default=0.0, help='fraction of validation set to hold out for final eval')
parser.add_argument('--initial_lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.98, help='momentum for SGD')
parser.add_argument('--nesterov', action='store_true', help='use Nesterov momentum')
parser.add_argument('--bootstrapping', action='store_true', help='use bootstrapping')
parser.add_argument('--map_optimizer', action='store_true', help='use MAP optimizer instead of MLE')

args = parser.parse_args()

# Random seed
seed = args.seed
tf.random.set_seed(seed)
np.random.seed(seed)

model_type = 'CNN-LSTM'

# Embedding
max_features = 20000
maxlen = 100
embedding_size = 128

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70

# Training
out_folder = args.out_folder
os.makedirs(out_folder, exist_ok=True)
experiment_id = args.id
batch_size = args.batch_size
epochs = args.epochs
validation_split = args.validation_split
hold_out_validation_split = args.hold_out_validation_split
checkpointing = args.checkpointing
initial_lr = args.initial_lr
momentum = args.momentum
nesterov = args.nesterov
bootstrapping = args.bootstrapping
map_optimizer = args.map_optimizer

'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''

# Prepare model saving directory.
current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = os.path.join(os.getcwd(), out_folder, current_date + f'_{experiment_id}')
model_name = f'{experiment_id}_imdb_{model_type}'
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

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# Split the training data into a training and a validation set
if validation_split > 0 or bootstrapping:
    train_indices, val_indices = split_dataset(x_train.shape[0], validation_split, bootstrap=bootstrapping, random=True)
    x_train, x_val = x_train[train_indices], x_train[val_indices]
    y_train, y_val = y_train[train_indices], y_train[val_indices]
    if hold_out_validation_split > 0.0:
        holdout_size = int(x_val.shape[0] * hold_out_validation_split)
        holdout_indices = np.random.choice(val_indices, holdout_size, replace=False)
        val_indices = np.setdiff1d(val_indices, holdout_indices)
        x_val_holdout, y_val_holdout = x_train[holdout_indices], y_train[holdout_indices]
        x_val, y_val = x_train[val_indices], y_train[val_indices]
        configuration['holdout_indices'] = holdout_indices.tolist()
        # Make sure the three sets are disjoint
        assert len(np.intersect1d(train_indices, val_indices)) == 0
        assert len(np.intersect1d(train_indices, holdout_indices)) == 0
        assert len(np.intersect1d(val_indices, holdout_indices)) == 0
    configuration['train_indices'] = train_indices.tolist()
    configuration['val_indices'] = val_indices.tolist()
else:
    x_val, y_val = x_test, y_test
    print('Using test set as validation set')
    if checkpointing:
        print("WARNING! YOU ARE VALIDATING ON THE TEST SET AND CHECKPOINTING IS ENABLED! SELECTION BIAS")
        sys.exit(1)

print(len(x_train), 'train sequences')
print(len(x_val), 'validation sequences')
print(len(x_test), 'test sequences')
print('x_train shape:', x_train.shape)
print('x_val shape:', x_val.shape)
print('x_test shape:', x_test.shape)

# Save configuration to json
fn = os.path.join(save_dir, model_name + '_config.json')
with open(fn, 'w') as f:
    json.dump(configuration, f, indent=4)

if map_optimizer:
    reg_weight = 1.0 / x_train.shape[0]
    print('Using MAP optimizer with reg_weight: ', str(reg_weight))
    pfac = priorfactory.GaussianPriorFactory(prior_stddev=1.0, weight=reg_weight)
    model = Sequential()
    model.add(pfac(Embedding(max_features, embedding_size, input_length=maxlen)))
    model.add(Dropout(0.25))
    model.add(pfac(Conv1D(filters,
                          kernel_size,
                          padding='valid',
                          activation='relu',
                          strides=1)))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(pfac(LSTM(lstm_output_size)))
    model.add(pfac(Dense(1)))
    model.add(Activation('sigmoid'))

else:
    print('Using MLE optimizer')
    model = Sequential()
    model.add(Embedding(max_features, embedding_size, input_length=maxlen))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

optimizer_ = SGD(learning_rate=initial_lr, momentum=momentum, nesterov=nesterov)

model.compile(loss='binary_crossentropy',
              optimizer=optimizer_,
              metrics=['accuracy'])
#model.summary()
print(model_type)

# Prepare callbacks for model saving and for learning rate adjustment.
callbacks = []
if checkpointing:
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_accuracy',
                                 save_weights_only=True,
                                 verbose=0,
                                 save_best_only=True)
    callbacks.append(checkpoint)

tqdm_callback = TqdmCallback(verbose=0)

callbacks.append(tqdm_callback)

history = model.fit(x_train, y_train,
    batch_size=batch_size,
    validation_data=(x_val, y_val),
    epochs=epochs, verbose=0,
    callbacks=callbacks)

if checkpointing:
    # Load best model weights (already saved)
    model.load_weights(filepath)
    print('Best model loaded from epoch: ', np.argmax(history.history['val_accuracy']) + 1)
else:
    # Save the model
    model.save(filepath)

# Score trained model.
score, acc = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score)
print('Test accuracy:', acc)
val_score, val_acc = model.evaluate(x_val, y_val, verbose=0)
print('Val score:', val_score)
print('Val accuracy:', val_acc)
scores = {'test_loss': score, 'test_accuracy': acc, 'val_loss': val_score, 'val_accuracy': val_acc, 'history': history.history}
if hold_out_validation_split > 0:
    holdout_score, holdout_acc = model.evaluate(x_val_holdout, y_val_holdout, verbose=0)
    print('Holdout score:', holdout_score)
    print('Holdout accuracy:', holdout_acc)
    scores['holdout_loss'] = holdout_score
    scores['holdout_accuracy'] = holdout_acc
# Save as dictionary
fn = os.path.join(save_dir, model_name + '_scores.json')
# Change all np.float32 to float
for k, v in scores['history'].items():
    scores['history'][k] = [float(x) for x in v]
with open(fn, 'w') as f:
    json.dump(scores, f, indent=4)

# Save predictions
y_pred = model.predict(x_test)
fn = os.path.join(save_dir, model_name + '_test_predictions.pkl')
with open(fn, 'wb') as f:
    pickle.dump(y_pred, f)

if validation_split > 0 or bootstrapping:
    y_pred = model.predict(x_val)
    fn = os.path.join(save_dir, model_name + '_val_predictions.pkl')
    with open(fn, 'wb') as f:
        pickle.dump(y_pred, f)

    if hold_out_validation_split > 0:
        y_pred = model.predict(x_val_holdout)
        fn = os.path.join(save_dir, model_name + '_val_holdout_predictions.pkl')
        with open(fn, 'wb') as f:
            pickle.dump(y_pred, f)
