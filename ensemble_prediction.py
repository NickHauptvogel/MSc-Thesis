import numpy as np
import tensorflow as tf
from keras.datasets import imdb, cifar10
import os
import json
import matplotlib.pyplot as plt
import pickle
import argparse
import sys
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MajorityVoteBounds.NeurIPS2020.optimize import optimize


def get_prediction(y_pred, y_test, indices, weights, num_classes):
    # y_pred has format (test_samples, models, classes)
    subset_y_pred = y_pred[:, indices, :]
    # Get mean prediction
    subset_y_pred_ensemble = np.average(subset_y_pred, axis=1, weights=weights)
    # Majority voting (mode of the predictions)
    if num_classes == 1:
        subset_y_pred = subset_y_pred[:, :, 0]  # Just to remove the last dimension
        subset_y_pred_ensemble = subset_y_pred_ensemble[:, 0]  # Just to remove the last dimension

        subset_y_pred_argmax_per_model = (subset_y_pred > 0.5).astype(int)
        subset_y_pred_softmax_average = (subset_y_pred_ensemble > 0.5).astype(int)
        ensemble_loss = tf.keras.losses.BinaryCrossentropy()(y_test, subset_y_pred_ensemble).numpy()
    else:
        subset_y_pred_argmax_per_model = np.argmax(subset_y_pred, axis=2)
        subset_y_pred_softmax_average = np.argmax(subset_y_pred_ensemble, axis=1)
        ensemble_loss = tf.keras.losses.CategoricalCrossentropy()(tf.one_hot(y_test, num_classes),
                                                                  subset_y_pred_ensemble).numpy()

    subset_y_pred_maj_vote = np.array(
        [np.argmax(np.bincount(subset_y_pred_argmax_per_model[i, :], weights=weights)) for i in
         range(subset_y_pred_argmax_per_model.shape[0])], dtype=int)
    ensemble_acc_maj_vote = np.mean(subset_y_pred_maj_vote == y_test)
    ensemble_acc_softmax_average = np.mean(subset_y_pred_softmax_average == y_test)

    return ensemble_acc_maj_vote, ensemble_acc_softmax_average, ensemble_loss


def load_all_predictions(folder: str, max_ensemble_size: int, test_pred_file_name='test_predictions.pkl', all_pred_file_name='all_predictions.pkl'):
    # Check whether predictions are already saved
    if not os.path.exists(os.path.join(folder, all_pred_file_name)):
        # Get all subdirectories
        subdirs = sorted([f.path for f in os.scandir(folder) if f.is_dir()])
        # Load the models
        accs = []
        losses = []
        predictions = []
        for subdir in subdirs:
            # Find prediction files
            pred_files = sorted([f.path for f in os.scandir(subdir) if f.name.endswith(test_pred_file_name)])
            if len(pred_files) == 0:
                print(f'No predictions found in {subdir}')
                continue
            # Find the score file
            score_file = [f.path for f in os.scandir(subdir) if f.name.endswith('scores.json')][0]
            # Load the scores
            with open(score_file, 'r') as f:
                scores = json.load(f)

            if isinstance(scores['test_accuracy'], list) and len(pred_files) != len(scores['test_accuracy']):
                print(f'Number of predictions and scores does not match in {subdir}')
                sys.exit(1)

            for i, pred_file in enumerate(pred_files):
                print(pred_file)
                # Load the predictions
                with open(pred_file, 'rb') as f:
                    y_pred = pickle.load(f)
                predictions.append(y_pred)
                if isinstance(scores['test_accuracy'], list):
                    accs.append(scores['test_accuracy'][i])
                    losses.append(scores['test_loss'][i])
                else:
                    accs.append(scores['test_accuracy'])
                    losses.append(scores['test_loss'])

                if len(predictions) == max_ensemble_size:
                    break
            if len(predictions) == max_ensemble_size:
                break

        # Concatenate all predictions to single array in the dimensions (test_samples, models, classes)
        y_pred = np.array(predictions).transpose(1, 0, 2)

        # Save the predictions of all models as well as models
        with open(os.path.join(folder, all_pred_file_name), 'wb') as f:
            pickle.dump(y_pred, f)
        with open(os.path.join(folder, 'accs.pkl'), 'wb') as f:
            pickle.dump(accs, f)
        with open(os.path.join(folder, 'losses.pkl'), 'wb') as f:
            pickle.dump(losses, f)

    else:
        # Load the predictions
        with open(os.path.join(folder, all_pred_file_name), 'rb') as f:
            y_pred = pickle.load(f)
        with open(os.path.join(folder, 'accs.pkl'), 'rb') as f:
            accs = pickle.load(f)
        with open(os.path.join(folder, 'losses.pkl'), 'rb') as f:
            losses = pickle.load(f)

    return y_pred, accs, losses


def ensemble_prediction(folder: str, max_ensemble_size: int, checkpoints_per_model:int, pac_bayes: bool, plot: bool, use_case: str, reps: int, include_lam: bool, tta: bool = False):

    num_independent_models = max_ensemble_size // checkpoints_per_model

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

    # Load the predictions
    y_pred, accs, losses = load_all_predictions(folder, max_ensemble_size)

    if tta:
        y_pred_tta, _, _ = load_all_predictions(folder, max_ensemble_size, 'test_tta_predictions.pkl', 'all_tta_predictions.pkl')

    results = {}
    categories = ['uniform_last_per_model']
    if tta:
        categories.append('uniform_tta_last_per_model')
    if checkpoints_per_model > 1:
        categories.append('uniform_all_per_model')
    if pac_bayes:
        categories.append('tnd_last_per_model')
        if include_lam:
            categories.append('lam_last_per_model')
        if checkpoints_per_model > 1:
            categories.append('tnd_all_per_model')
            if include_lam:
                categories.append('lam_all_per_model')

    for category in categories:

        print('Category:', category)

        ensemble_accs_mean = []
        ensemble_accs_std = []
        ensemble_losses_mean = []
        ensemble_losses_std = []
        for ensemble_size in range(2, num_independent_models + 1):
            ensemble_accs_maj_vote = []
            ensemble_accs_softmax_average = []
            ensemble_losses = []
            for i in tqdm(range(reps)):

                # Choose randomly ensemble_size integers from 0 to num_independent_models
                indices = np.random.choice(num_independent_models, ensemble_size, replace=False)

                if 'last_per_model' in category:
                    indices = [i * checkpoints_per_model + checkpoints_per_model - 1 for i in indices]
                elif 'all_per_model' in category:
                    indices = [i * checkpoints_per_model + j for i in indices for j in range(checkpoints_per_model)]

                weights = None
                if 'tnd' in category or 'lam' in category:
                    rhos, pac_results = optimize(use_case, len(indices), 'DUMMY', 'iRProp', 1, 'DUMMY', folder, False, indices=indices)
                    # Get the weights
                    if 'tnd' in category:
                        weights = rhos[1]
                    elif 'lam' in category:
                        weights = rhos[0]

                if 'tta' in category:
                    y_pred_ = y_pred_tta
                else:
                    y_pred_ = y_pred

                ensemble_acc_maj_vote, ensemble_acc_softmax_average, ensemble_loss = get_prediction(y_pred_, y_test, indices, weights, num_classes)
    
                ensemble_accs_maj_vote.append(ensemble_acc_maj_vote)
                ensemble_accs_softmax_average.append(ensemble_acc_softmax_average)
                ensemble_losses.append(ensemble_loss)

            ensemble_accs_mean.append((ensemble_size, np.mean(ensemble_accs_maj_vote), np.mean(ensemble_accs_softmax_average)))
            ensemble_accs_std.append((ensemble_size, np.std(ensemble_accs_maj_vote), np.std(ensemble_accs_softmax_average)))
            print('Mean Accuracy Majority Vote:', np.round(np.mean(ensemble_accs_maj_vote), 3), '(Softmax Average:', np.round(np.mean(ensemble_accs_softmax_average), 3), ') with', ensemble_size, 'models')
            ensemble_losses_mean.append((ensemble_size, np.mean(ensemble_losses)))
            ensemble_losses_std.append((ensemble_size, np.std(ensemble_losses)))
    
    
        ensemble_accs_mean = np.array(ensemble_accs_mean)
        ensemble_accs_std = np.array(ensemble_accs_std)
        
        results[category] = (ensemble_accs_mean, ensemble_accs_std, ensemble_losses_mean, ensemble_losses_std)

    # Save results
    with open(os.path.join(folder, 'ensemble_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # Plot the results
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8, zorder=0)
    ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    
    for category in categories:
        ensemble_accs_mean, ensemble_accs_std, _, _ = results[category]
    
        #plt.plot(ensemble_accs_mean[:, 0], ensemble_accs_mean[:, 1], label='Mean accuracy majority vote')
        plt.plot(ensemble_accs_mean[:, 0], ensemble_accs_mean[:, 2], label=f'{category}')
        # Std as area around the mean
        #plt.fill_between(ensemble_accs_mean[:, 0], ensemble_accs_mean[:, 1] - ensemble_accs_std[:, 1],
        #                 ensemble_accs_mean[:, 1] + ensemble_accs_std[:, 1], alpha=0.3, label='±1σ majority vote')
        plt.fill_between(ensemble_accs_mean[:, 0], ensemble_accs_mean[:, 2] - ensemble_accs_std[:, 2],
                         ensemble_accs_mean[:, 2] + ensemble_accs_std[:, 2], alpha=0.3)
    plt.xlabel('Ensemble size')
    plt.ylabel('Accuracy')
    plt.ylim(ylim)
    # Horizontal line for the accuracy of the best model
    plt.axhline(max(accs), color='orange', linestyle='--', label='Best individual model')
    # Horizontal line for accuracy of Wen et al. (2020), interpolated from the figure at 0.9363
    plt.axhline(wenzeL_acc, color='grey', linestyle='--', label='Wenzel et al. (2020)')
    plt.title('Mean ensemble accuracy (softmax average)')
    plt.xticks(np.arange(ensemble_accs_mean[0][0], ensemble_accs_mean[-1][0] + 1, 4))
    plt.grid()
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(folder, 'ensemble_accs.pdf'))
    if plot:
        plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8, zorder=0)
    ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    ax.minorticks_on()

    for category in categories:
        _, _, ensemble_losses_mean, ensemble_losses_std = results[category]

        plt.plot(*zip(*ensemble_losses_mean), label=f'({category})')
        # Std as area around the mean
        plt.fill_between(np.array(ensemble_losses_mean)[:, 0], np.array(ensemble_losses_mean)[:, 1] - np.array(ensemble_losses_std)[:, 1],
                         np.array(ensemble_losses_mean)[:, 1] + np.array(ensemble_losses_std)[:, 1], alpha=0.3)
    # Horizontal line for the loss of the best model
    plt.axhline(min(losses), color='orange', linestyle='--', label='Best individual model')
    # Horizontal line for loss of Wen et al. (2020), interpolated from the figure at 0.217
    plt.axhline(wenzeL_loss, color='grey', linestyle='--', label='Wenzel et al. (2020)')
    plt.xlabel('Ensemble size')
    plt.ylabel('Categorical cross-entropy')
    plt.title('Mean ensemble loss')
    plt.xticks(np.arange(ensemble_losses_mean[0][0], ensemble_losses_mean[-1][0] + 1, 2))
    plt.grid()
    # Legend upper right
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(folder, 'ensemble_losses.pdf'))
    if plot:
        plt.show()


    return results

def main():
    # Configuration
    parser = argparse.ArgumentParser(description='Ensemble prediction')
    parser.add_argument('--folder', type=str, default='ResNet20_CIFAR/results/10_checkp_every_40_wenzel_0_2_val',
                        help='Folder with the models')
    parser.add_argument('--max_ensemble_size', type=int, default=50,
                        help='Maximum ensemble size')
    parser.add_argument('--checkpoints_per_model', type=int, default=1, help='Number of checkpoints per independent model')
    parser.add_argument('--pac-bayes', action='store_true', help='Use pac-bayes weights')
    parser.add_argument('--plot', action='store_true', help='Plot the results')
    parser.add_argument('--use_case', type=str, default='cifar10')
    parser.add_argument('--reps', type=int, help='Number of repetitions', required=False, default=5)
    parser.add_argument('--include_lam', action='store_true', help='Include lambda in the ensemble')
    parser.add_argument('--tta', action='store_true', help='Use test time augmentation predictions')

    args = parser.parse_args()
    folder = args.folder
    max_ensemble_size = args.max_ensemble_size
    checkpoints_per_model = args.checkpoints_per_model
    pac_bayes = args.pac_bayes
    plot = args.plot
    use_case = args.use_case
    reps = args.reps
    include_lam = args.include_lam

    folder = 'CNN-LSTM_IMDB/results/50_independent_wenzel_0_2_hold_out_val'
    max_ensemble_size = 50
    checkpoints_per_model = 1
    pac_bayes = True
    use_case = 'imdb'
    reps = 1

    folder = 'ResNet20_CIFAR/results/50_independent_wenzel_no_checkp_no_val'
    max_ensemble_size = 50
    checkpoints_per_model = 1
    pac_bayes = False
    use_case = 'cifar10'
    reps = 5
    tta = True


    ensemble_prediction(folder, max_ensemble_size, checkpoints_per_model, pac_bayes, plot, use_case, reps, include_lam, tta)


if __name__ == '__main__':
    main()