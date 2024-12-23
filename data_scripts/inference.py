from keras.models import load_model
from utils import load_dir, segmap_to_image
import os
import numpy as np
from custom_metrics import DiceCoefficient
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Preprocess dataset for training, validation, and testing.")
parser.add_argument(
    "--model_path", 
    type=str, 
    required=True, 
    help="Path to the model, should end in .keras"
)

parser.add_argument(
    "--data_dir", 
    type=str, 
    default=os.path.join(os.getcwd(), "data", "test"),
    required=False, 
    help="Path to the directory to be segmented, defaults to data/test/"
)

args = parser.parse_args()

model_path = args.model_path
data_dir = args.data_dir
output_dir = os.path.join(os.getcwd(), "inferenced_data")

num_classes = 8
color_mapping = {
    0: (128, 0, 0),   # Building
    1: (128, 64, 128), # Road
    2: (192, 0, 192),  # Static Car
    3: (0, 128, 0),    # Tree
    4: (128, 128, 0),  # Low Vegetation
    5: (64, 64, 0),    # Human
    6: (64, 0, 128),   # Moving Car
    7: (0, 0, 0),      # Background Clutter
}

os.makedirs(output_dir, exist_ok=True)
test_images, _ = load_dir(data_dir, num_classes, color_mapping)

# Get metrics
dice_coef = DiceCoefficient()
#load model
model = load_model(model_path, custom_objects={"DiceCoefficient": dice_coef})

# get order files will be processed in (since load_dir() sorts how os.listdir() does)
input_file_names:list = os.listdir(os.path.join(data_dir, "images"))

# inference each image individually
for i, image in enumerate(test_images):
    expanded_img = np.expand_dims(image, axis=0)
    segmap = model.predict(expanded_img)
    segmap_to_image(segmap, color_mapping, output_dir, filename=f"{input_file_names[i].split('.')[0]}_label.png")
