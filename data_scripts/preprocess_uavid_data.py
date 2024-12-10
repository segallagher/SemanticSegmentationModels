import os
import argparse
from utils import process_directory

# Set up argument parsing
parser = argparse.ArgumentParser(description="Preprocess dataset for training, validation, and testing.")
parser.add_argument(
    "--dataset_dir", 
    type=str, 
    required=True, 
    help="Path to the directory of the dataset containing the train, val, and test sudirectories (e.g., 'uavid_v1.5_official_release_image')"
)

# Parse arguments
args = parser.parse_args()

dataset_dir = args.dataset_dir
train_dir = os.path.join(dataset_dir, "uavid_train")
val_dir = os.path.join(dataset_dir, "uavid_val")
test_dir = os.path.join(dataset_dir, "uavid_test")
size = [256,256]
color_channels = 3

base_dir = os.getcwd()
prepocessed_dir = os.path.join(base_dir, "data")

print("Processing", dataset_dir)
process_directory(input_directory=train_dir, output_directory=os.path.join(prepocessed_dir, "train"))
process_directory(input_directory=test_dir, output_directory=os.path.join(prepocessed_dir, "test"))
process_directory(input_directory=val_dir, output_directory=os.path.join(prepocessed_dir, "val"))
