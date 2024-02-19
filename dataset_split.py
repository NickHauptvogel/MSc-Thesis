import numpy as np


def split_dataset(ensemble_size, train_samples, validation_split, random=True, bootstrap=False):

    for i in range(1, ensemble_size+1):
        if random:
            indices = np.random.choice(train_samples, train_samples, replace=bootstrap)
        else:
            indices = np.arange(train_samples)

        # Split the indices into training and validation
        if bootstrap:
            train_indices = indices
            # Validation indices are the remaining indices
            validation_indices = np.setdiff1d(np.arange(train_samples), train_indices)
        else:
            split = int(np.floor(validation_split * train_samples))
            train_indices = indices[:-split]
            validation_indices = indices[-split:]

        # Save the indices
        # TODO


if __name__ == '__main__':
    ensemble_size = 5
    train_samples = 100
    validation_split = 0.2
    split_dataset(ensemble_size, train_samples, validation_split, bootstrap=False)
    split_dataset(ensemble_size, train_samples, validation_split, bootstrap=True)