import os
from utils import process_directory

dataset_dir = os.path.join("E:", "datasets", "uavid_v1.5_official_release_image", "uavid_v1.5_official_release_image")
train_dir = os.path.join(dataset_dir, "uavid_train")
val_dir = os.path.join(dataset_dir, "uavid_val")
test_dir = os.path.join(dataset_dir, "uavid_test")
size = [256,256]
color_channels = 3

base_dir = os.getcwd()
prepocessed_dir = os.path.join(base_dir, "data")

process_directory(input_directory=train_dir, output_directory=os.path.join(prepocessed_dir, "train"))
process_directory(input_directory=test_dir, output_directory=os.path.join(prepocessed_dir, "test"))
process_directory(input_directory=val_dir, output_directory=os.path.join(prepocessed_dir, "val"))
