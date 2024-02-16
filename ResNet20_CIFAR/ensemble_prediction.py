import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.models import load_model
import os
import json
import matplotlib.pyplot as plt
import pickle
import argparse

# Configuration
parser = argparse.ArgumentParser(description='Ensemble prediction')
parser.add_argument('--folder', type=str, default='results/30_independent_wenzel_hyperparams_checkp/',
                    help='Folder with the models')
parser.add_argument('--max_ensemble_size', type=int, default=30,
                    help='Maximum ensemble size')
parser.add_argument('--plot', action='store_true', help='Plot the results')

args = parser.parse_args()
folder = args.folder
max_ensemble_size = args.max_ensemble_size
plot = args.plot

num_classes = 10
# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_test -= x_train_mean

print(x_test.shape[0], 'test samples')

# Check whether predictions are already saved
if not os.path.exists(os.path.join(folder, 'all_predictions.pkl')):
    # Get all subdirectories
    subdirs = [f.path for f in os.scandir(folder) if f.is_dir()]
    # Shuffle the subdirectories
    np.random.shuffle(subdirs)
    # Load the models
    models = []
    accs = []
    losses = []
    for subdir in subdirs:
        # Find the model file
        model_file = [f.path for f in os.scandir(subdir) if f.name.endswith('.h5')][0]
        # Load the model
        model = load_model(model_file)
        # Find the score file
        score_file = [f.path for f in os.scandir(subdir) if f.name.endswith('scores.json')][0]
        # Load the scores
        with open(score_file, 'r') as f:
            scores = json.load(f)
            test_acc = scores['test_accuracy']
            test_loss = scores['test_loss']
        models.append(model)
        accs.append(test_acc)
        losses.append(test_loss)
        if len(models) == max_ensemble_size:
            break

    # Predict
    y_pred = np.zeros((x_test.shape[0], len(models), num_classes), dtype=np.float32)
    for i, model in enumerate(models):
        print('Predicting with model', i+1, 'of', len(models))
        y_pred[:, i, :] = model.predict(x_test)

    # Save the predictions of all models as well as models
    with open(os.path.join(folder, 'all_predictions.pkl'), 'wb') as f:
        pickle.dump(y_pred, f)
    with open(os.path.join(folder, 'accs.pkl'), 'wb') as f:
        pickle.dump(accs, f)
    with open(os.path.join(folder, 'losses.pkl'), 'wb') as f:
        pickle.dump(losses, f)

else:
    # Load the predictions
    with open(os.path.join(folder, 'all_predictions.pkl'), 'rb') as f:
        y_pred = pickle.load(f)
    with open(os.path.join(folder, 'accs.pkl'), 'rb') as f:
        accs = pickle.load(f)
    with open(os.path.join(folder, 'losses.pkl'), 'rb') as f:
        losses = pickle.load(f)

ensemble_accs_mean = []
ensemble_accs_std = []
ensemble_losses_mean = []
ensemble_losses_std = []
for ensemble_size in range(2, max_ensemble_size + 1):
    ensemble_accs = []
    ensemble_losses = []
    for i in range(20):
        # Choose randomly ensemble_size integers from 0 to len(models)
        indices = np.random.choice(len(accs), ensemble_size, replace=False)
        subset_y_pred = y_pred[:, indices, :]
        # Get mean prediction
        subset_y_pred_ensemble = np.mean(subset_y_pred, axis=1)
        # Get majority vote
        subset_y_pred_argmax = np.argmax(subset_y_pred, axis=2)
        # Majority voting (mode of the predictions)
        subset_y_pred_vote = np.array([np.argmax(np.bincount(subset_y_pred_argmax[i, :])) for i in range(subset_y_pred_argmax.shape[0])], dtype=int)

        # Evaluate the predictions with accuracy
        ensemble_acc = np.mean(subset_y_pred_vote == y_test[:, 0])
        ensemble_accs.append(ensemble_acc)
        ensemble_loss = tf.keras.losses.CategoricalCrossentropy()(tf.one_hot(y_test[:, 0], num_classes), subset_y_pred_ensemble).numpy()
        ensemble_losses.append(ensemble_loss)

    ensemble_accs_mean.append((ensemble_size, np.mean(ensemble_accs)))
    ensemble_accs_std.append((ensemble_size, np.std(ensemble_accs)))
    print('Mean Accuracy:', np.round(np.mean(ensemble_accs), 3), 'with', ensemble_size, 'models')
    ensemble_losses_mean.append((ensemble_size, np.mean(ensemble_losses)))
    ensemble_losses_std.append((ensemble_size, np.std(ensemble_losses)))
    print('Mean Loss:', np.round(np.mean(ensemble_losses), 3), 'with', ensemble_size, 'models')

# Save the results
with open(os.path.join(folder, 'ensemble_accs_mean.pkl'), 'wb') as f:
    pickle.dump(ensemble_accs_mean, f)
with open(os.path.join(folder, 'ensemble_accs_std.pkl'), 'wb') as f:
    pickle.dump(ensemble_accs_std, f)
with open(os.path.join(folder, 'ensemble_losses_mean.pkl'), 'wb') as f:
    pickle.dump(ensemble_losses_mean, f)
with open(os.path.join(folder, 'ensemble_losses_std.pkl'), 'wb') as f:
    pickle.dump(ensemble_losses_std, f)

# Plot the results
plt.figure(figsize=(6, 4))
plt.plot(*zip(*ensemble_accs_mean), label='Mean accuracy')
# Std as area around the mean
plt.fill_between(np.array(ensemble_accs_mean)[:, 0], np.array(ensemble_accs_mean)[:, 1] - np.array(ensemble_accs_std)[:, 1],
                 np.array(ensemble_accs_mean)[:, 1] + np.array(ensemble_accs_std)[:, 1], alpha=0.3, label='±1σ')
plt.xlabel('Ensemble size')
plt.ylabel('Accuracy')
plt.ylim(0.9, 0.95)
# Horizontal line for the accuracy of the best model
plt.axhline(max(accs), color='orange', linestyle='--', label='Best individual model')
# Horizontal line for accuracy of Wen et al. (2020), interpolated from the figure at 0.9363
plt.axhline(0.9363, color='grey', linestyle='--', label='Wenzel et al. (2020)')
plt.title('Ensemble accuracy')
plt.xticks(np.arange(ensemble_accs_mean[0][0], ensemble_accs_mean[-1][0] + 1, 2))
plt.grid()
# Legend lower right
plt.legend(loc='lower right')
plt.savefig(os.path.join(folder, 'ensemble_accs.pdf'))
if plot:
    plt.show()

plt.figure(figsize=(6, 4))
plt.plot(*zip(*ensemble_losses_mean), label='Mean loss')
# Std as area around the mean
plt.fill_between(np.array(ensemble_losses_mean)[:, 0], np.array(ensemble_losses_mean)[:, 1] - np.array(ensemble_losses_std)[:, 1],
                 np.array(ensemble_losses_mean)[:, 1] + np.array(ensemble_losses_std)[:, 1], alpha=0.3, label='±1σ')
# Horizontal line for the loss of the best model
plt.axhline(min(losses), color='orange', linestyle='--', label='Best individual model')
# Horizontal line for loss of Wen et al. (2020), interpolated from the figure at 0.217
plt.axhline(0.217, color='grey', linestyle='--', label='Wenzel et al. (2020)')
plt.xlabel('Ensemble size')
plt.ylabel('Categorical cross-entropy')
plt.title('Ensemble loss')
plt.xticks(np.arange(ensemble_losses_mean[0][0], ensemble_losses_mean[-1][0] + 1, 2))
plt.grid()
# Legend upper right
plt.legend(loc='upper right')
plt.savefig(os.path.join(folder, 'ensemble_losses.pdf'))
if plot:
    plt.show()
