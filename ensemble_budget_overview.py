import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import argparse
from ensemble_prediction import ensemble_prediction


# Configuration
parser = argparse.ArgumentParser(description='Ensemble prediction')
parser.add_argument('--folder', type=str, default='ResNet20_CIFAR/results/epoch_budget_2',
                    help='Folder with the models')
parser.add_argument('--max_ensemble_size', type=int, default=15,
                    help='Maximum ensemble size')
parser.add_argument('--plot', action='store_true', help='Plot the results')
parser.add_argument('--use_case', type=str, default='cifar10')

args = parser.parse_args()
folder = args.folder
max_ensemble_size = args.max_ensemble_size
plot = args.plot
use_case = args.use_case

if use_case == 'cifar10':
    wenzeL_acc = 0.9363
    wenzeL_loss = 0.217
    ylim_acc = (0.9, 0.95)
    ylim_loss = (0.15, 0.5)
elif use_case == 'imdb':
    wenzeL_acc = 0.8703
    wenzeL_loss = 0.3044
    ylim_acc = (0.8, 0.88)
    ylim_loss = (0.2, 0.4)
else:
    raise ValueError('Unknown use case')

results_last = {}

for i in range(2, max_ensemble_size + 1):

    subdir = os.path.join(folder, f"{i:02d}")
    res = ensemble_prediction(subdir, i, 1, False, False, use_case, 1)

    for category in res.keys():
        if category not in results_last:
            results_last[category] = ([], [], [], [])

        (mean, std, l_mean, l_std) = res['uniform_last_per_model']
        results_last[category][0].append(mean[-1])
        results_last[category][1].append(std[-1])
        results_last[category][2].append(l_mean[-1])
        results_last[category][3].append(l_std[-1])

# Plot the results
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.grid(which='major', color='#DDDDDD', linewidth=0.8, zorder=0)
ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
ax.minorticks_on()

for category in results_last.keys():
    ensemble_accs_mean, ensemble_accs_std, _, _ = results_last[category]

    ensemble_accs_mean = np.array(ensemble_accs_mean)
    ensemble_accs_std = np.array(ensemble_accs_std)

    #plt.plot(ensemble_accs_mean[:, 0], ensemble_accs_mean[:, 1], label='Mean accuracy majority vote')
    plt.plot(ensemble_accs_mean[:, 0], ensemble_accs_mean[:, 2], label=f'{category}')
    # Std as area around the mean
    #plt.fill_between(ensemble_accs_mean[:, 0], ensemble_accs_mean[:, 1] - ensemble_accs_std[:, 1],
    #                 ensemble_accs_mean[:, 1] + ensemble_accs_std[:, 1], alpha=0.3, label='±1σ majority vote')
    plt.fill_between(ensemble_accs_mean[:, 0], ensemble_accs_mean[:, 2] - ensemble_accs_std[:, 2],
                     ensemble_accs_mean[:, 2] + ensemble_accs_std[:, 2], alpha=0.3,)
    plt.xlabel('Ensemble size')
    plt.ylabel('Accuracy')
    plt.ylim(ylim_acc)
    # Horizontal line for accuracy of Wen et al. (2020), interpolated from the figure at 0.9363
    plt.axhline(wenzeL_acc, color='grey', linestyle='--', label='Wenzel et al. (2020)')
    plt.xticks(np.arange(ensemble_accs_mean[0][0], ensemble_accs_mean[-1][0] + 1, 2))
    plt.title('Mean epoch budget ensemble accuracy (softmax average)')
    plt.grid()
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(folder, 'ensemble_accs.pdf'))
    if plot:
        plt.show()

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.grid(which='major', color='#DDDDDD', linewidth=0.8, zorder=0)
ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
ax.minorticks_on()

for category in results_last.keys():
    _, _, ensemble_losses_mean, ensemble_losses_std = results_last[category]

    plt.plot(*zip(*ensemble_losses_mean), label=f'({category})')
    # Std as area around the mean
    plt.fill_between(np.array(ensemble_losses_mean)[:, 0], np.array(ensemble_losses_mean)[:, 1] - np.array(ensemble_losses_std)[:, 1],
                     np.array(ensemble_losses_mean)[:, 1] + np.array(ensemble_losses_std)[:, 1], alpha=0.3)
    # Horizontal line for loss of Wen et al. (2020), interpolated from the figure at 0.217
    plt.axhline(wenzeL_loss, color='grey', linestyle='--', label='Wenzel et al. (2020)')
    plt.xlabel('Ensemble size')
    plt.ylabel('Categorical cross-entropy')
    plt.title('Mean epoch budget ensemble loss')
    plt.xticks(np.arange(ensemble_losses_mean[0][0], ensemble_losses_mean[-1][0] + 1, 2))
    plt.ylim(ylim_loss)
    plt.grid()
    # Legend upper right
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(folder, 'ensemble_losses.pdf'))
    if plot:
        plt.show()
