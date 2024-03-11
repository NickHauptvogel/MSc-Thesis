import numpy as np


# Default values for ResNets from the paper
def sse_lr_schedule(epoch, B=200, M=5, initial_lr=0.2):
    """Learning Rate Schedule for ResNet models
    """
    ceil = np.ceil(B / M)
    lr = (initial_lr / 2) * (np.cos(np.pi * ((epoch) % ceil) / ceil) + 1)
    print(f'Epoch {epoch}, LR: {lr}')
    return lr


if __name__ == '__main__':
    # Example usage
    for epoch in range(200):
        sse_lr_schedule(epoch)