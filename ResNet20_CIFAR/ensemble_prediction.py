import numpy as np
from keras.datasets import cifar10
from keras.models import load_model
import os
import json
import matplotlib.pyplot as plt
import pickle

# Configuration
max_ensemble_size = 30
folder = 'results/30_independent/'

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
if not os.path.exists(os.path.join(folder, 'all_predictions.npy')):
    # Get all subdirectories
    subdirs = [f.path for f in os.scandir(folder) if f.is_dir()]
    # Shuffle the subdirectories
    np.random.shuffle(subdirs)
    # Load the models
    models = []
    accs = []
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
        models.append(model)
        accs.append(test_acc)
        if len(models) == max_ensemble_size:
            break

    # Predict
    y_pred = np.zeros((x_test.shape[0], len(models)), dtype=int)
    for i, model in enumerate(models):
        print('Predicting with model', i+1, 'of', len(models))
        y_pred[:, i] = np.argmax(model.predict(x_test, verbose=1), axis=1)

    # Save the predictions of all models as well as models
    with open(os.path.join(folder, 'all_predictions.pkl'), 'wb') as f:
        pickle.dump(y_pred, f)
    with open(os.path.join(folder, 'accs.pkl'), 'wb') as f:
        pickle.dump(accs, f)

else:
    # Load the predictions
    with open(os.path.join(folder, 'all_predictions.pkl'), 'rb') as f:
        y_pred = pickle.load(f)
    with open(os.path.join(folder, 'accs.pkl'), 'rb') as f:
        accs = pickle.load(f)

ensemble_accs = []
ensemble_fractions = []
for ensemble_size in range(2, max_ensemble_size + 1):
    # Choose randomly ensemble_size integers from 0 to len(models)
    indices = np.random.choice(len(accs), ensemble_size, replace=False)
    subset_y_pred = y_pred[:, indices]
    # Majority voting (mode of the predictions)
    subset_y_pred = np.array([np.argmax(np.bincount(subset_y_pred[i, :])) for i in range(subset_y_pred.shape[0])])

    # Evaluate the predictions with accuracy
    ensemble_acc = np.mean(subset_y_pred == y_test[:, 0])
    ensemble_accs.append((ensemble_size, ensemble_acc))
    print('Accuracy:', ensemble_acc, 'with', ensemble_size, 'models')
    ensemble_better = np.mean(ensemble_acc > [accs[i] for i in indices])
    ensemble_fractions.append((ensemble_size, ensemble_better))
    # Fraction of members where the ensemble is better
    print('Fraction of ensemble members where the ensemble is better:', ensemble_better)

# Plot the results
plt.figure()
plt.plot(*zip(*ensemble_accs))
plt.xlabel('Ensemble size')
plt.ylabel('Accuracy')
plt.title('Ensemble accuracy')
plt.savefig(os.path.join(folder, 'ensemble_accs.pdf'))
plt.show()

plt.figure()
plt.plot(*zip(*ensemble_fractions))
plt.xlabel('Ensemble size')
plt.ylabel('Fraction')
plt.title('Fraction of ensemble members where the ensemble is better')
plt.savefig(os.path.join(folder, 'ensemble_fractions.pdf'))
plt.show()


