'''Train a recurrent convolutional network on the IMDB sentiment
classification task.

Gets to 0.8498 test accuracy after 2 epochs. 41s/epoch on K520 GPU.
'''
from __future__ import print_function

import keras
import tensorflow as tf
import numpy as np
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import SGD
from keras.datasets import imdb
import argparse
import os
from tqdm.keras import TqdmCallback
from datetime import datetime
import json

import priorfactory


parser = argparse.ArgumentParser()
parser.add_argument('--id', type=int, default=1, help='ID of the experiment')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--validation_split', type=float, default=0.2, help='validation split')
parser.add_argument('--checkpointing', action='store_true', help='save the best model during training')
parser.add_argument('--initial_lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.98, help='momentum for SGD')
parser.add_argument('--nesterov', action='store_true', help='use Nesterov momentum')

args = parser.parse_args()

# Random seed
seed = 1
tf.random.set_seed(seed)

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
map_regularization = True
experiment_id = args.id
batch_size = args.batch_size
epochs = args.epochs
validation_split = args.validation_split
checkpointing = args.checkpointing
initial_lr = args.initial_lr
momentum = args.momentum
nesterov = True #args.nesterov

'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''

# Prepare model saving directory.
current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = os.path.join(os.getcwd(), 'results', current_date + f'_{experiment_id:02d}')
model_name = f'{experiment_id:02d}_cifar10_{model_type}'
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
# Save configuration to json
fn = os.path.join(save_dir, model_name + '_config.json')
with open(fn, 'w') as f:
    json.dump(configuration, f, indent=4)

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# Split the training data into a training and a validation set
# Not random, as the original paper by Wenzel et al.
if validation_split > 0:
    indices = np.arange(x_train.shape[0])
    split = int(x_train.shape[0] * (1 - validation_split))
    x_train, x_val = x_train[:split], x_train[split:]
    y_train, y_val = y_train[:split], y_train[split:]
else:
    x_val, y_val = x_test, y_test
    print('Using test set as validation set')

print(len(x_train), 'train sequences')
print(len(x_val), 'validation sequences')
print(len(x_test), 'test sequences')
print('x_train shape:', x_train.shape)
print('x_val shape:', x_val.shape)
print('x_test shape:', x_test.shape)

reg_weight = 1.0 / x_train.shape[0]
pfac = priorfactory.GaussianPriorFactory(prior_stddev=1.0,
                                             weight=reg_weight)

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
else:
    # Save the model
    model.save(filepath)

# Score trained model.
score, acc = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score)
print('Test accuracy:', acc)
scores = {'test_loss': score, 'test_accuracy': acc, 'history': history.history}
# Save as dictionary
fn = os.path.join(save_dir, model_name + '_scores.json')
# Change all np.float32 to float
for k, v in scores['history'].items():
    scores['history'][k] = [float(x) for x in v]
with open(fn, 'w') as f:
    json.dump(scores, f, indent=4)