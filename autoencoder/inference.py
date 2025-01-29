from keras.models import load_model
from utils import load_dir, segmap_to_image, get_hyperparam
import os
import numpy as np
from custom_metrics import DiceCoefficient
import argparse
import time
import json
from pathlib import Path

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


# Get hyperparams
hyperparam = get_hyperparam()

# Get directories
model_path = Path(args.model_path).resolve()

data_dir = Path(hyperparam["data_path"]).resolve() / "test"

output_dir = "output_dir"
if hyperparam.get("output_dir"):
    output_dir = hyperparam["output_dir"]
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Initialize inference_metrics dict 
inference_metrics = {}

# Get model size on disk
inference_metrics["disk_size"] = model_path.stat().st_size

# Data info
hyperparam = get_hyperparam()

# Get images from test directory
test_images, _ = load_dir(data_dir, hyperparam["num_classes"], hyperparam["reverse_color_mapping"])

# Get metrics
dice_coef = DiceCoefficient()
# Load model
model = load_model(model_path, custom_objects={"DiceCoefficient": dice_coef})

# get order files will be processed in (since load_dir() sorts how os.listdir() does)
input_file_names:list = os.listdir(data_dir / "images")

# Create inference directory
inference_dir = output_dir / "inferenced_data"
inference_dir.mkdir(parents=True, exist_ok=True)


# Get inference times and create inference images
inference_times=[]
for i, image in enumerate(test_images):
    expanded_img = np.expand_dims(image, axis=0)
    start_time = time.time_ns()
    segmap = model.predict(expanded_img, verbose=0)
    stop_time = time.time_ns()
    inference_times.append(stop_time-start_time)
    segmap_to_image(segmap, hyperparam["reverse_color_mapping"], inference_dir, filename=f"{input_file_names[i].split('.')[0]}_label.png")
inference_metrics["inference_times"] = inference_times
inference_metrics["avg_inference_time"] = np.average(np.array(inference_times))

# Output metrics
with open(output_dir / "inference_metrics.json", 'w') as f:
    json.dump(inference_metrics, f, indent=4)