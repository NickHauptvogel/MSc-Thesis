import os


def sort_out_files():
    target_folder = 'CNN-LSTM_IMDB/results/50_independent_smalllr_bootstr_hold_out_val'
    out_files_folder = 'CNN-LSTM_IMDB'
    #target_folder = 'ResNet20_CIFAR/results/50_independent_wenzel_no_checkp_bootstr'
    #out_files_folder = 'ResNet20_CIFAR'

    # Get all .out files in out_files_folder
    out_files = [f for f in os.listdir(out_files_folder) if f.endswith('.out')]
    # Get all subdirectories in folder
    subdirs = [f.path for f in os.scandir(target_folder) if f.is_dir()]

    for out_file in out_files:
        print(out_file)
        # Get the experiment id
        experiment_id = int(out_file.split('_')[-1].replace('.out', ''))
        # Find the subdirectory with the same experiment id
        subdir = [f for f in subdirs if f.endswith(f'{experiment_id:02d}')]
        if len(subdir) == 0:
            # If the experiment id is not found, try again with a single digit
            subdir = [f for f in subdirs if f.endswith(f'{experiment_id}')]
        subdir = subdir[0]
        # Move the .out file to the subdirectory
        os.rename(os.path.join(out_files_folder, out_file), os.path.join(subdir, out_file))


def sort_epoch_budget_folders():
    target_folder = 'ResNet20_CIFAR/results/epoch_budget'
    current_folder = 'ResNet20_CIFAR/results/epoch_budget'

    # Get all subdirectories in current_folder that start with 2024
    subdirs = [f.path for f in os.scandir(current_folder) if f.is_dir() and f.name.startswith('2024')]

    for subdir in subdirs:
        # Get the ensemble size from the subdirectory name
        ensemble = subdir.split('_')[-2]
        model = subdir.split('_')[-1]
        # Make sure ensemble folder exists in target_folder
        ensemble_folder = os.path.join(target_folder, ensemble)
        if not os.path.exists(ensemble_folder):
            os.makedirs(ensemble_folder)
        # Move the subdirectory to the ensemble folder
        os.rename(subdir, os.path.join(ensemble_folder, model))


def rename_files():
    folder = 'ResNet20_CIFAR/results/50_independent_wenzel_no_checkp_bootstr'
    # Get all files that end with predictions.pkl recursively
    files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(folder) for f in filenames if
             f.endswith('predictions.pkl')]
    files = [f for f in files if 'all' not in f]
    for file in files:
        # Rename to test_predictions.pkl
        new_name = file.replace('predictions.pkl', 'test_predictions.pkl')
        os.rename(file, new_name)
        print(f'{file} -> {new_name}')


if __name__ == '__main__':
    sort_out_files()
    #sort_epoch_budget_folders()
