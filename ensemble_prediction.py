import numpy as np
import tensorflow as tf
from keras.datasets import imdb, cifar10
import os
import json
import matplotlib.pyplot as plt
import pickle
import argparse


def ensemble_prediction(folder: str, max_ensemble_size: int, plot: bool, use_case: str):

    if use_case=='cifar10':
        wenzeL_acc = 0.9363
        wenzeL_loss = 0.217
        ylim = (0.9, 0.95)
        num_classes = 10
        _, (_, y_test) = cifar10.load_data()
        y_test = y_test[:, 0]

    elif use_case=='imdb':
        wenzeL_acc = 0.8703
        wenzeL_loss = 0.3044
        ylim = (0.83, 0.88)
        num_classes = 1
        max_features = 20000
        _, (_, y_test) = imdb.load_data(num_words=max_features)

    else:
        raise ValueError('Unknown use case')


    # Check whether predictions are already saved
    if not os.path.exists(os.path.join(folder, 'all_predictions.pkl')):
        # Get all subdirectories
        subdirs = [f.path for f in os.scandir(folder) if f.is_dir()]
        # Shuffle the subdirectories
        np.random.shuffle(subdirs)
        # Load the models
        accs = []
        losses = []
        predictions = []
        for subdir in subdirs:
            # Find prediction file
            pred_file = [f.path for f in os.scandir(subdir) if f.name.endswith('test_predictions.pkl')]
            if len(pred_file) == 0:
                print(f'No predictions found in {subdir}')
                continue
            pred_file = pred_file[0]
            # Load the predictions
            with open(pred_file, 'rb') as f:
                y_pred = pickle.load(f)
            predictions.append(y_pred)
            # Find the score file
            score_file = [f.path for f in os.scandir(subdir) if f.name.endswith('scores.json')][0]
            # Load the scores
            with open(score_file, 'r') as f:
                scores = json.load(f)
                test_acc = scores['test_accuracy']
                test_loss = scores['test_loss']
            accs.append(test_acc)
            losses.append(test_loss)
            if len(predictions) == max_ensemble_size:
                break

        # Concatenate all predictions to single array in the dimensions (test_samples, models, classes)
        y_pred = np.array(predictions).transpose(1, 0, 2)


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
        ensemble_accs_maj_vote = []
        ensemble_accs_ensemble_argmax = []
        ensemble_losses = []
        for i in range(20):
            # Choose randomly ensemble_size integers from 0 to len(models)
            indices = np.random.choice(len(accs), ensemble_size, replace=False)
            # y_pred has format (test_samples, models, classes)
            subset_y_pred = y_pred[:, indices, :]
            # Get mean prediction
            subset_y_pred_ensemble = np.mean(subset_y_pred, axis=1)
            # Majority voting (mode of the predictions)
            if num_classes == 1:
                subset_y_pred = subset_y_pred[:, :, 0]
                subset_y_pred_ensemble = subset_y_pred_ensemble[:, 0]
                subset_y_pred_argmax = (subset_y_pred > 0.5).astype(int)
                subset_y_pred_ensemble_argmax = (subset_y_pred_ensemble > 0.5).astype(int)
                ensemble_loss = tf.keras.losses.BinaryCrossentropy()(y_test, subset_y_pred_ensemble).numpy()
            else:
                subset_y_pred_argmax = np.argmax(subset_y_pred, axis=2)
                subset_y_pred_ensemble_argmax = np.argmax(subset_y_pred_ensemble, axis=1)
                ensemble_loss = tf.keras.losses.CategoricalCrossentropy()(tf.one_hot(y_test, num_classes),
                                                                          subset_y_pred_ensemble).numpy()

            subset_y_pred_maj_vote = np.array([np.argmax(np.bincount(subset_y_pred_argmax[i, :])) for i in range(subset_y_pred_argmax.shape[0])], dtype=int)
            ensemble_acc_maj_vote = np.mean(subset_y_pred_maj_vote == y_test)
            ensemble_acc_ensemble_argmax = np.mean(subset_y_pred_ensemble_argmax == y_test)

            ensemble_accs_maj_vote.append(ensemble_acc_maj_vote)
            ensemble_accs_ensemble_argmax.append(ensemble_acc_ensemble_argmax)
            ensemble_losses.append(ensemble_loss)


        ensemble_accs_mean.append((ensemble_size, np.mean(ensemble_accs_maj_vote), np.mean(ensemble_accs_ensemble_argmax)))
        ensemble_accs_std.append((ensemble_size, np.std(ensemble_accs_maj_vote), np.std(ensemble_accs_ensemble_argmax)))
        print('Mean Accuracy Majority Vote:', np.round(np.mean(ensemble_accs_maj_vote), 3), '(Argmax Ensemble:', np.round(np.mean(ensemble_accs_ensemble_argmax), 3), ') with', ensemble_size, 'models')
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

    ensemble_accs_mean = np.array(ensemble_accs_mean)
    ensemble_accs_std = np.array(ensemble_accs_std)

    # Plot the results
    plt.figure(figsize=(6, 4))
    plt.plot(ensemble_accs_mean[:, 0], ensemble_accs_mean[:, 1], label='Mean accuracy majority vote')
    plt.plot(ensemble_accs_mean[:, 0], ensemble_accs_mean[:, 2], label='Mean accuracy argmax ensemble')
    # Std as area around the mean
    plt.fill_between(ensemble_accs_mean[:, 0], ensemble_accs_mean[:, 1] - ensemble_accs_std[:, 1],
                     ensemble_accs_mean[:, 1] + ensemble_accs_std[:, 1], alpha=0.3, label='±1σ majority vote')
    plt.fill_between(ensemble_accs_mean[:, 0], ensemble_accs_mean[:, 2] - ensemble_accs_std[:, 2],
                     ensemble_accs_mean[:, 2] + ensemble_accs_std[:, 2], alpha=0.3, label='±1σ argmax ensemble')
    plt.xlabel('Ensemble size')
    plt.ylabel('Accuracy')
    plt.ylim(ylim)
    # Horizontal line for the accuracy of the best model
    plt.axhline(max(accs), color='orange', linestyle='--', label='Best individual model')
    # Horizontal line for accuracy of Wen et al. (2020), interpolated from the figure at 0.9363
    plt.axhline(wenzeL_acc, color='grey', linestyle='--', label='Wenzel et al. (2020)')
    plt.title('Ensemble accuracy')
    plt.xticks(np.arange(ensemble_accs_mean[0][0], ensemble_accs_mean[-1][0] + 1, 4))
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
    plt.axhline(wenzeL_loss, color='grey', linestyle='--', label='Wenzel et al. (2020)')
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

if __name__ == '__main__':
    # Configuration
    parser = argparse.ArgumentParser(description='Ensemble prediction')
    parser.add_argument('--folder', type=str, default='CNN-LSTM_IMDB/results/50_independent_smalllr_bootstr_hold_out_val',
                        help='Folder with the models')
    parser.add_argument('--max_ensemble_size', type=int, default=50,
                        help='Maximum ensemble size')
    parser.add_argument('--plot', action='store_true', help='Plot the results')
    parser.add_argument('--use_case', type=str, default='imdb')

    args = parser.parse_args()
    folder = args.folder
    max_ensemble_size = args.max_ensemble_size
    plot = args.plot
    use_case = args.use_case

    ensemble_prediction(folder, max_ensemble_size, plot, use_case)