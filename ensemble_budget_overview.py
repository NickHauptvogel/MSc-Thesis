import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import argparse


# Configuration
parser = argparse.ArgumentParser(description='Ensemble prediction')
parser.add_argument('--folder', type=str, default='ResNet20_CIFAR/results/epoch_budget_no_checkp',
                    help='Folder with the models')
parser.add_argument('--max_ensemble_size', type=int, default=20,
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

ensemble_accs_mean = []
ensemble_accs_std = []
ensemble_losses_mean = []
ensemble_losses_std = []
for subdir in [os.path.join(folder, str(i)) for i in range(2, max_ensemble_size + 1)]:
    # Load ensemble_accs and ensemble_losses
    with open(os.path.join(subdir, 'ensemble_accs_mean.pkl'), 'rb') as f:
        mean = pickle.load(f)
    with open(os.path.join(subdir, 'ensemble_accs_std.pkl'), 'rb') as f:
        std = pickle.load(f)
    with open(os.path.join(subdir, 'ensemble_losses_mean.pkl'), 'rb') as f:
        l_mean = pickle.load(f)
    with open(os.path.join(subdir, 'ensemble_losses_std.pkl'), 'rb') as f:
        l_std = pickle.load(f)

    # Take last value of mean and std for each ensemble size
    ensemble_accs_mean.append(mean[-1])
    # Standard deviation will be almost 0 (as only max ensemble size is considered, so always the same models)
    ensemble_accs_std.append(std[-1])

    ensemble_losses_mean.append(l_mean[-1])
    ensemble_losses_std.append(l_std[-1])

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
plt.ylim(ylim_acc)
# Horizontal line for accuracy of Wen et al. (2020), interpolated from the figure at 0.9363
plt.axhline(wenzeL_acc, color='grey', linestyle='--', label='Wenzel et al. (2020)')
# Integers on x-axis
plt.xticks(np.arange(ensemble_accs_mean[0][0], ensemble_accs_mean[-1][0] + 1, 2))
plt.title('Epoch budget ensemble accuracy')
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
# Horizontal line for loss of Wen et al. (2020), interpolated from the figure at 0.217
plt.axhline(wenzeL_loss, color='grey', linestyle='--', label='Wenzel et al. (2020)')
plt.xlabel('Ensemble size')
plt.ylabel('Categorical cross-entropy')
plt.title('Epoch budget ensemble loss')
plt.xticks(np.arange(ensemble_losses_mean[0][0], ensemble_losses_mean[-1][0] + 1, 2))
plt.ylim(ylim_loss)
plt.grid()
# Legend upper right
plt.legend(loc='upper right')
plt.savefig(os.path.join(folder, 'ensemble_losses.pdf'))
if plot:
    plt.show()
