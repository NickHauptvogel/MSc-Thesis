import os


folder = 'CNN-LSTM_IMDB/results/30_independent_wenzel_hyperparams_random_val'
out_files_folder = 'CNN-LSTM_IMDB'

# Get all .out files in out_files_folder
out_files = [f for f in os.listdir(out_files_folder) if f.endswith('.out')]
# Get all subdirectories in folder
subdirs = [f.path for f in os.scandir(folder) if f.is_dir()]

for out_file in out_files:
    # Get the experiment id
    experiment_id = int(out_file.split('_')[1].replace('.out', ''))
    # Find the subdirectory with the same experiment id
    subdir = [f for f in subdirs if f.endswith(f'_{experiment_id:02d}')][0]
    # Move the .out file to the subdirectory
    os.rename(os.path.join(out_files_folder, out_file), os.path.join(subdir, out_file))