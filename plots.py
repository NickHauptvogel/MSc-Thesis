import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import os

def plot_lr_loss(outpath):

    # Find scores file
    scores_file = [f for f in os.listdir(outpath) if f.endswith('scores.json')][0]
    with open(outpath+scores_file, 'r') as f:
        scores = json.load(f)

    loss = scores['history']['loss']
    val_loss = scores['history']['val_loss']
    acc = scores['history']['accuracy']
    val_acc = scores['history']['val_accuracy']
    lr = scores['history']['lr']
    epochs = range(1, len(loss) + 1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(epochs, loss, label='Training loss')
    ax.plot(epochs, val_loss, label='Validation loss')
    ax.plot(epochs, 30*np.array(lr), label='Learning rate (scaled)', c=(0.1, 0.1, 0.1, 0.5))
    ax.set_title('Training and validation loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.ylim(0, 6)
    plt.savefig(outpath + 'loss.pdf')
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(epochs, acc, label='Train accuracy')
    ax.plot(epochs, val_acc, label='Val accuracy')
    ax.plot(epochs, 5*np.array(lr), label='Learning rate (scaled)', c=(0.1, 0.1, 0.1, 0.5))
    ax.set_title('Training and validation accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend()
    plt.ylim(0, 1)
    plt.savefig(outpath + 'accuracy.pdf')
    plt.show()

def plot_pac_bayes(outpath):
    # Load the data
    rhos_file = [f for f in os.listdir(outpath) if f.endswith('rhos.csv')][0]
    risks_file = [f for f in os.listdir(outpath) if f.endswith('bootstrap-iRProp.csv')][0]
    df_rhos = pd.read_csv(outpath + rhos_file, sep=';')
    df_risks = pd.read_csv(outpath + risks_file, sep=';')

    print("number of val points: ", df_risks['n_min'][0])
    print("number of tandem val points: ", df_risks['n2_min'][0])

    # 3 subplots
    fig, ax = plt.subplots(3, 1, figsize=(15, 10))
    # Plot the risk for each member in subplot 1 as bar plot without spacing between the bars
    ax[0].bar(df_rhos['h'], df_rhos['risk'], label='Risk', color='black', width=1.05)
    # Plot the weights
    ax[1].bar(df_rhos['h'], df_rhos['rho_lam'], label='rho_lam', color='green')
    ax[2].bar(df_rhos['h'], df_rhos['rho_mv2'], label='rho_mv2', color='blue')

    # x axis label for last subplot
    ax[2].set_xlabel('Ensemble member')
    ax[0].set_ylabel('Risk')
    ax[1].set_ylabel('rho')
    ax[2].set_ylabel('rho')

    for a in ax:
        a.set_xticks(df_rhos['h'])
        a.legend(loc='upper left')

    plt.suptitle('Risk and weights for each ensemble member')
    plt.tight_layout()
    plt.savefig(outpath + 'risk_weights.pdf')
    plt.show()

    # Plot the bounds for the different ensemble types
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8, zorder=0)
    # Show the minor grid as well. Style it in very light gray as a thin,
    # dotted line.
    ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    # Make the minor ticks and gridlines show.
    ax.minorticks_on()
    # Plot the bounds for the different ensemble types next to each other
    ax.bar([1 - 0.2, 2 - 0.2, 3 - 0.2], [df_risks['unf_pbkl'][0], df_risks['lam_pbkl'][0], df_risks['tnd_pbkl'][0]],
           label='FO bound', color=(0.2, 0.6, 0.2, 0.5), width=0.15, edgecolor=(0.2, 0.6, 0.2, 1.), linewidth=1,
           zorder=3)
    ax.bar([1, 2, 3], [df_risks['unf_ctd'][0], df_risks['lam_ctd'][0], df_risks['tnd_ctd'][0]], label='CTD bound',
           color=(0.1, 0.1, 0.1, 0.5), width=0.15, edgecolor=(0.1, 0.1, 0.1, 1.), linewidth=1, zorder=3)
    ax.bar([1 + 0.2, 2 + 0.2, 3 + 0.2], [df_risks['unf_tnd'][0], df_risks['lam_tnd'][0], df_risks['tnd_tnd'][0]],
           label='TND bound', color=(0., 0., 1., 0.5), width=0.15, edgecolor=(0., 0., 1., 1.), linewidth=1,
           zorder=3)

    max_y = max([df_risks['unf_pbkl'][0], df_risks['lam_pbkl'][0], df_risks['tnd_pbkl'][0], df_risks['unf_ctd'][0],
                 df_risks['lam_ctd'][0], df_risks['tnd_ctd'][0], df_risks['unf_tnd'][0], df_risks['lam_tnd'][0],
                 df_risks['tnd_tnd'][0]])
    ax.set_ylim(0, max_y * 1.1)

    # Three horizontal lines for the risk of the different ensemble types
    # ax.axhline(df_risks['unf_mv_risk_softmax_avg'][0], color=(0.2, 0.6, 0.2, 0.5),  label='risk ρ*_u', zorder=5, linestyle="dotted")
    # ax.axhline(df_risks['lam_mv_risk_softmax_avg'][0], color=(0.1, 0.1, 0.1, 0.5),  label='risk ρ*_FO', zorder=5, linestyle="dotted")
    # ax.axhline(df_risks['tnd_mv_risk_softmax_avg'][0], color=(0., 0., 1., 0.5), label='risk ρ*_TND', zorder=5, linestyle="dotted")
    # ax.axhline(df_risks['unf_mv_risk_maj_vote'][0], color=(0.2, 0.6, 0.2, 0.5), label='risk ρ*_u', zorder=5)
    # ax.axhline(df_risks['lam_mv_risk_maj_vote'][0], color=(0.1, 0.1, 0.1, 0.5), label='risk ρ*_FO', zorder=5)
    # ax.axhline(df_risks['tnd_mv_risk_maj_vote'][0], color=(0., 0., 1., 0.5), label='risk ρ*_TND', zorder=5)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Uniform', 'Lambda', 'Tandem'])
    ax.set_xlabel('Weighting scheme')
    ax.set_ylabel('Risk')
    ax.set_title('Risk of different ensemble weightings')
    ax.legend(loc='upper right')
    plt.savefig(outpath + 'risk_bounds.pdf')
    plt.show()

    # Plot the risks for the different ensemble types
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8, zorder=0)
    # Show the minor grid as well. Style it in very light gray as a thin,
    # dotted line.
    ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    # Make the minor ticks and gridlines show.
    ax.minorticks_on()
    # Plot the bounds for the different ensemble types next to each other
    ax.bar([1 + 0.1, 2 + 0.1, 3 + 0.1],
           [1-df_risks['unf_mv_risk_softmax_avg'][0], 1-df_risks['lam_mv_risk_softmax_avg'][0],
            1-df_risks['tnd_mv_risk_softmax_avg'][0]],
           label='Softmax average', color=(0.2, 0.6, 0.2, 0.5), width=0.15, edgecolor=(0.2, 0.6, 0.2, 1.),
           linewidth=1,
           zorder=3)
    ax.bar([1 - 0.1, 2 - 0.1, 3 - 0.1], [1-df_risks['unf_mv_risk_maj_vote'][0], 1-df_risks['lam_mv_risk_maj_vote'][0],
                                         1-df_risks['tnd_mv_risk_maj_vote'][0]],
           label='Majority vote', color=(0., 0., 1., 0.5), width=0.19, edgecolor=(0., 0., 1., 1.), linewidth=1,
           zorder=3)

    ax.set_ylim(0.5, 1.0)
    min_uniform = min([df_risks['unf_mv_risk_softmax_avg'][0], df_risks['unf_mv_risk_maj_vote'][0]])
    min_lambda = min([df_risks['lam_mv_risk_softmax_avg'][0], df_risks['lam_mv_risk_maj_vote'][0]])
    min_tandem = min([df_risks['tnd_mv_risk_softmax_avg'][0], df_risks['tnd_mv_risk_maj_vote'][0]])
    print("minimal risk uniform: ", min_uniform, " (accuracy: ", 1 - min_uniform, ")")
    print("minimal risk lambda: ", min_lambda, " (accuracy: ", 1 - min_lambda, ")")
    print("minimal risk tandem: ", min_tandem, " (accuracy: ", 1 - min_tandem, ")")
    # Label bars with risks
    ax.text(1.15, 1 - df_risks['unf_mv_risk_softmax_avg'][0], f"{1-df_risks['unf_mv_risk_softmax_avg'][0]:.3f}", ha='center', va='bottom')
    ax.text(2.15, 1 - df_risks['lam_mv_risk_softmax_avg'][0], f"{1-df_risks['lam_mv_risk_softmax_avg'][0]:.3f}", ha='center', va='bottom')
    ax.text(3.15, 1 - df_risks['tnd_mv_risk_softmax_avg'][0], f"{1-df_risks['tnd_mv_risk_softmax_avg'][0]:.3f}", ha='center', va='bottom')
    ax.text(0.85, 1 - df_risks['unf_mv_risk_maj_vote'][0], f"{1-df_risks['unf_mv_risk_maj_vote'][0]:.3f}", ha='center', va='bottom')
    ax.text(1.85, 1 - df_risks['lam_mv_risk_maj_vote'][0], f"{1-df_risks['lam_mv_risk_maj_vote'][0]:.3f}", ha='center', va='bottom')
    ax.text(2.85, 1 - df_risks['tnd_mv_risk_maj_vote'][0], f"{1-df_risks['tnd_mv_risk_maj_vote'][0]:.3f}", ha='center', va='bottom')


    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Uniform', 'Lambda', 'Tandem'])
    ax.set_xlabel('Weighting scheme')
    ax.set_ylabel('Risk')
    ax.set_title('Risk of different ensemble weightings')
    ax.legend(loc='lower right')
    plt.savefig(outpath + 'risks.pdf')
    plt.show()


if __name__ == '__main__':
    #plot_lr_loss('ResNet20_CIFAR/results/10_sse_every_40_wenzel_0_2_val/20240312_080615_01/')
    #plot_lr_loss('ResNet20_CIFAR/results/10_checkp_every_40_wenzel_0_2_val/20240312_102307_01/')
    plot_pac_bayes('ResNet20_CIFAR/results/cifar100/resnet110/10_checkp_every_40_wenzel_0_2_val/pac-bayes/')
    #plot_lr_loss('ResNet20_CIFAR/results/50_independent_wenzel_no_checkp_val/20240227_162729_01/')
    #plot_pac_bayes('ResNet20_CIFAR/results/50_independent_wenzel_no_checkp_val/pac-bayes/')
    #plot_pac_bayes('ResNet20_CIFAR/results/10_sse_every_40_wenzel_0_2_val/pac-bayes/')