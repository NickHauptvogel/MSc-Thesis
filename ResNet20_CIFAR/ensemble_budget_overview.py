import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

folder = '.'
ensemble_accs_mean = []
ensemble_accs_std = []
ensemble_losses_mean = []
ensemble_losses_std = []
for subdir in ['./' + str(i) for i in range(2, 21)]:
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

# Plot the results
plt.figure(figsize=(6, 4))
plt.plot(*zip(*ensemble_accs_mean), label='Mean accuracy')
# Std as area around the mean
plt.fill_between(np.array(ensemble_accs_mean)[:, 0], np.array(ensemble_accs_mean)[:, 1] - np.array(ensemble_accs_std)[:, 1],
                 np.array(ensemble_accs_mean)[:, 1] + np.array(ensemble_accs_std)[:, 1], alpha=0.3, label='±1σ')
plt.xlabel('Ensemble size')
plt.ylabel('Accuracy')
plt.ylim(0.9, 0.95)
# Horizontal line for accuracy of Wen et al. (2020), interpolated from the figure at 0.9363
plt.axhline(0.9363, color='grey', linestyle='--', label='Wenzel et al. (2020)')
# Integers on x-axis
plt.xticks(np.arange(ensemble_accs_mean[0][0], ensemble_accs_mean[-1][0] + 1, 2))
plt.title('Epoch budget ensemble accuracy')
plt.grid()
# Legend lower right
plt.legend(loc='lower right')
plt.savefig(os.path.join(folder, 'ensemble_accs.pdf'))
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(*zip(*ensemble_losses_mean), label='Mean loss')
# Std as area around the mean
plt.fill_between(np.array(ensemble_losses_mean)[:, 0], np.array(ensemble_losses_mean)[:, 1] - np.array(ensemble_losses_std)[:, 1],
                 np.array(ensemble_losses_mean)[:, 1] + np.array(ensemble_losses_std)[:, 1], alpha=0.3, label='±1σ')
# Horizontal line for loss of Wen et al. (2020), interpolated from the figure at 0.217
plt.axhline(0.217, color='grey', linestyle='--', label='Wenzel et al. (2020)')
plt.xlabel('Ensemble size')
plt.ylabel('Categorical cross-entropy')
plt.title('Epoch budget ensemble loss')
plt.xticks(np.arange(ensemble_losses_mean[0][0], ensemble_losses_mean[-1][0] + 1, 2))
plt.ylim(0.15, 0.5)
plt.grid()
# Legend upper right
plt.legend(loc='upper right')
plt.savefig(os.path.join(folder, 'ensemble_losses.pdf'))
plt.show()
