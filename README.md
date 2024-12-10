Run this project in a Linux environment for most compatability \
If on Windows, just run this in WSL2. Its not worth it trying to get tensorflow-gpu working. \
If you want to train using a GPU, you must have a CUDA supported GPU and install the correct version of `CUDA` and `cudnn` installed before running the installation steps.

## Requirements
- Python
- Pip

## Installation
1. Create a python virtual environment with the command \
`python3 -m venv .venv`
2. Activate virtual environment with the command \
`source .venv/bin/activate`
3. Run `install.sh`

## Running
1. Download your dataset
2. Preprocess your dataset, currently only provides a script to preprocess the uavid dataset (available at https://uavid.nl/) \
Run the python script `preprocess_uavid_data.py` in the `data_scripts` directory through command line with argument `--dataset_dir` dataset_dir should be the folder containing the subfolders `uavid_train`, `uavid_val`, `uavid_test`
3. Run your model \
    3.1 Explanation \
    Each model architecture has a folder of the same name (IE: the autoencoder model would be in the autoencoder directory) \
\
    3.2 Modify Hyperparameters
    In the model architecture folder, modify the `hyperparameters.json` file \
\
    3.3 Run Model Training
    In the model architecture folder, run the python file of the architectures name (IE: the file to train the autoencoder in the autoencoder directory is `autoencoder.py`)

## Output Models
Models are saved to a file with the name \<architecture\>.keras \
In order to run the models, you must pass it the custom objects that are passed to it as in file `data_script/inference.py`.

## Inferencing Your Model
Inferencing your model can be done by running the `data_scripts/inference.py` script through the commmand line \
The following are the command line arguments for the script \
--model_path (path to your model) REQUIRED \
--data_dir (directory where your data is)
