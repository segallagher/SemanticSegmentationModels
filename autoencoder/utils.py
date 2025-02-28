from tensorflow.keras import callbacks, backend as K, metrics
from keras import layers, models
from pathlib import Path
from PIL import Image
import numpy as np
import json
from io import StringIO
import sys
import re
import wandb
import time

# Hyperparam
def get_hyperparam():
    curr_dir = Path(__file__).resolve().parent

    # Get Hyperparameters
    hyperparamPath = curr_dir / "hyperparameters.json"
    hyperparam = None

    try:
        with open(hyperparamPath, 'r') as file:
            # Load Hyperparameters
            hyperparam = json.load(file)
            # Convert json encoded map to python
            hyperparam["color_mapping"] = {tuple(map(int, key.split(','))): value for key, value in hyperparam["color_mapping"].items()}
            hyperparam["reverse_color_mapping"] = {value: key for key, value in hyperparam["color_mapping"].items()}
            # Assign num_classes based off color_mapping
            hyperparam["num_classes"] = len(hyperparam["color_mapping"])
    except FileNotFoundError:
        print("Hyperparameters file not found")
    except json.JSONDecodeError:
        print("Error decoding hyperparameters from json")
    except Exception as e:
        print(f"Error when reading hyperparameters: {e}")
    return hyperparam

# Data loader
# Loads data from a file structure like the one found on UAVid dataset
def load_data(directory:Path, num_classes:int, color_to_class_map:dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_dir = directory / "train"
    val_dir = directory / "val"
    test_dir = directory / "test"

    train_images, train_labels = load_dir(train_dir, num_classes, color_to_class_map)
    val_images, val_labels = load_dir(val_dir, num_classes, color_to_class_map)
    test_images, _ = load_dir(test_dir, num_classes, color_to_class_map)
    return train_images, train_labels, val_images, val_labels, test_images

# loads a directory from the UAVid dataset
def load_dir(directory:Path, num_classes:int, color_to_class_map:dict) -> tuple[np.ndarray, np.ndarray]:
    img_dir = directory / "images"
    img_list = []
    for image in img_dir.iterdir():
        # Path to image
        img_path =img_dir / image
        with Image.open(img_path) as img:
            # Convert image to np array
            img_array = np.array(img)
            img_list.append(img_array)

    lab_dir = directory / "labels"
    lab_list = []
    if lab_dir.exists():
        for image in lab_dir.iterdir():
            # Path to image
            img_path = lab_dir / image
            with Image.open(img_path) as img:
                # Convert image to array
                img_array = np.array(img)

                # Make emtpy labeled image
                labeled_img = np.zeros(shape=(img_array.shape[0], img_array.shape[1], num_classes), dtype=np.uint8)

                # Start encoding masks of images
                for color, label in color_to_class_map.items():
                    # Generate mask of the current label
                    mask = np.all(img_array == np.array(color), axis=-1)

                    # One-hot encode
                    labeled_img[mask, label] = 1
                lab_list.append(labeled_img)
    
    img_arr = None
    if len(img_list) > 0:
        img_arr = np.stack(img_list)
    lab_arr = None
    if len(lab_list) > 0:
        lab_arr = np.stack(lab_list)
    return img_arr, lab_arr

# Metrics
class DiceCoefficient(metrics.Metric):
    def __init__(self, name='dice_coefficient', smooth=100, **kwargs):
        super(DiceCoefficient, self).__init__(name=name, **kwargs)
        self.smooth = smooth
        self.intersection = self.add_weight(name='intersection', initializer='zeros')
        self.true_sum = self.add_weight(name='true_sum', initializer='zeros')
        self.pred_sum = self.add_weight(name='pred_sum', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Flatten the tensors to 1D for calculation
        y_true_f = K.cast(K.flatten(y_true), dtype='float32')
        y_pred_f = K.cast(K.flatten(y_pred), dtype='float32')

        # Calculate intersection and sums for Dice coefficient
        intersection = K.sum(y_true_f * y_pred_f)
        true_sum = K.sum(y_true_f)
        pred_sum = K.sum(y_pred_f)
        
        # Update the state with current batch values
        self.intersection.assign_add(intersection)
        self.true_sum.assign_add(true_sum)
        self.pred_sum.assign_add(pred_sum)

    def result(self):
        # Compute the Dice coefficient
        dice = (2. * self.intersection + self.smooth) / (self.true_sum + self.pred_sum + self.smooth)
        return dice

    def reset_state(self):
        # Reset the weights at the start of each epoch
        self.intersection.assign(0.)
        self.true_sum.assign(0.)
        self.pred_sum.assign(0.)

class LogTrainingMetrics(callbacks.Callback):
    def __init__(self, monitor:str, additional_metrics:list=[], output_path:str="best_metrics.json",
                 total_param:int=0, trainable_param:int=0, non_train_param:int=0, memory:str=0,
                 ):
        super().__init__()
        # Get metric names
        self.monitor = monitor
        self.additional_metrics = additional_metrics
        # Initialize metric values
        self.best_epoch = None
        self.best_monitor_value = -float('inf')
        self.additional_metric_values = [-float('inf') for _ in additional_metrics]
        self.total_epoch = 0
        # Count Parameters
        self.total_param = total_param
        self.trainable_param = trainable_param
        self.non_train_param = non_train_param
        # Misc
        self.output_path = output_path
        self.memory = memory
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        # Update metric values if epoch's primary monitor value is higher than previous best_monitor_value
        cur_val = logs.get(self.monitor)
        if cur_val is not None and cur_val > self.best_monitor_value:
            self.best_monitor_value = cur_val
            self.best_epoch = epoch + 1
            # update additional metric values
            self.additional_metric_values = [logs.get(metric) for metric in self.additional_metrics]

        # Update epoch count
        self.total_epoch += 1

    def on_train_end(self, logs = None):
        if self.best_epoch is not None:
            # Create data log
            data = {
                "best_epoch": self.best_epoch,
                "total_epoch": self.total_epoch,
                self.monitor: self.best_monitor_value,
                "total_param": self.total_param,
                "trainable_param": self.trainable_param,
                "non_train_param": self.non_train_param,
                "memory_usage": self.memory,
            }
            # Add additional metrics
            for i, metric in enumerate(self.additional_metric_values):
                data[f"{self.additional_metrics[i]}"] = metric
            # Log Time
            self.total_time = time.time() - self.start_time
            data["time"] = self.total_time

            # Log to wandb
            for key in data:
                wandb.config[key] = data[key]

            # Write to json
            with open(self.output_path, 'w') as f:
                json.dump(data, f, indent=4)

def get_model_summary_string(model: models.Model) -> str:
    # Capture summary output
    string_io = StringIO()
    original_stdout = sys.stdout
    sys.stdout = string_io
    
    model.summary()

    # Restore original output
    sys.stdout = original_stdout

    # Convert summary output to string
    return string_io.getvalue()


def get_param_count(summary: str) -> tuple[int, int, int]:
    # Parse parameter counts
    total_param, trainable_param, non_train_param = 0, 0, 0
    total_param_match = re.search(r"Total params: \s*([0-9,]+)", summary)
    if total_param_match:
        total_param = int(total_param_match.group(1).replace(",",""))
    else:
        print("Total Params not found")

    trainable_param_match = re.search(r"Trainable params: \s*([0-9,]+)", summary)
    if trainable_param_match:
        trainable_param = int(trainable_param_match.group(1).replace(",",""))
    else:
        print("Trainable Params not found")

    non_train_param_match = re.search(r"Non-trainable params: \s*([0-9,]+)", summary)
    if non_train_param_match:
        non_train_param = int(non_train_param_match.group(1).replace(",",""))
    else:
        print("Non Trainable Params not found")
    return total_param, trainable_param, non_train_param

def get_mem_size(summary: str) -> int:
    
    total_mem = 0
    total_mem_match = re.search(r"Total params: [\s0-9,]*\(([0-9. A-Z]*)\)", summary)
    if total_mem_match:
        total_mem = total_mem_match.group(1)
    else:
        print("Total mem not found")
    return total_mem

# Inference

def segmap_to_image(segmaps:np.ndarray, class_to_color_map:dict, output_dir:str=Path.cwd(), color_channels:int=3, filename:str=None):
    for i, segmap in enumerate(segmaps):

        # Get the class with the highest probability
        argmax_labels = np.argmax(segmap, axis=-1)
        
        # convert labels to colors
        image_arr = np.zeros((segmap.shape[0],segmap.shape[1], color_channels), dtype=np.uint8)
        for label, color in class_to_color_map.items():
            image_arr[argmax_labels == label] = color

        # turn array into image
        image = Image.fromarray(image_arr)

        # Save image
        if filename:
            image.save(Path(output_dir) / filename)
        else:
            image.save(Path(output_dir) / f"{i}.png")
        