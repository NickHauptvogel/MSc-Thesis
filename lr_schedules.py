import numpy as np
import tensorflow as tf

def cifar_schedule(epoch, initial_lr, epochs):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs. Expressed relatively to 200 epochs as
    8/20, 12/20, 16/20, 18/20.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = initial_lr
    if epoch > int(0.9 * epochs):
        lr *= 0.5e-3
    elif epoch > int(0.8 * epochs):
        lr *= 1e-3
    elif epoch > int(0.6 * epochs):
        lr *= 1e-2
    elif epoch > int(0.4 * epochs):
        lr *= 1e-1
    return lr


# Default values for ResNets from the paper
def sse_lr_schedule(epoch, B=200, M=5, initial_lr=0.2):
    """Learning Rate Schedule for ResNet models
    """
    ceil = np.ceil(B / M)
    lr = (initial_lr / 2) * (np.cos(np.pi * ((epoch) % ceil) / ceil) + 1)
    print(f'Epoch {epoch}, LR: {lr}')
    return lr


# From https://github.com/google/uncertainty-baselines/blob/main/uncertainty_baselines/schedules.py
def step_decay_schedule(epoch, initial_lr, decay_ratio, decay_epochs, warmup_epochs):
    """Learning rate schedule.

    It starts with a linear warmup to the initial learning rate over
    `warmup_epochs`. This is found to be helpful for large batch size training
    (Goyal et al., 2018). The learning rate's value then uses the initial
    learning rate, and decays by a multiplier at the start of each epoch in
    `decay_epochs`. The stepwise decaying schedule follows He et al. (2015).
    """
    lr = initial_lr
    if warmup_epochs >= 1:
        lr *= epoch / warmup_epochs
    decay_epochs = [warmup_epochs] + decay_epochs
    for index, start_epoch in enumerate(decay_epochs):
        lr = tf.where(
            epoch >= start_epoch,
            initial_lr * decay_ratio ** index,
            lr)
    return lr
