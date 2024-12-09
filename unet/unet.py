from keras import losses, optimizers
from keras.metrics import OneHotMeanIoU
from keras.models import load_model, Functional
import tensorflow as tf
from pathlib import Path
from utils import load_data, create_unet, DiceCoefficient, LogBestEpoch
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import json

curr_dir = Path(__file__).resolve().parent

# Load Hyperparameters
hyperparamPath = curr_dir / "hyperparameters.json"
hyperparam = None

try:
    with open(hyperparamPath, 'r') as file:
        # Load Hyperparameters
        hyperparam = json.load(file)
        # Convert json encoded map to python
        hyperparam["color_mapping"] = converted_mapping = {tuple(map(int, key.split(','))): value for key, value in hyperparam["color_mapping"].items()}
        # Assign num_classes based off color_mapping
        hyperparam["num_classes"] = len(hyperparam["color_mapping"])
except FileNotFoundError:
    print("Hyperparameters file not found")
except json.JSONDecodeError:
    print("Error decoding hyperparameters from json")
except Exception as e:
    print(f"Error when reading hyperparameters: {e}")

# Load data
data_dir = None
if hyperparam.get("data_path"):
    data_dir = Path(curr_dir / hyperparam["data_path"]).resolve()
else:
    data_dir = curr_dir / "data"
train_images, train_labels, val_images, val_labels, test_images = load_data(data_dir, hyperparam["num_classes"], hyperparam["color_mapping"])

# Allow GPU memory growth to avoid running out of GPU memory
# code from https://www.tensorflow.org/guide/gpu
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# Create model metrics
meaniou_metric = OneHotMeanIoU(num_classes=hyperparam["num_classes"])
dice_coef = DiceCoefficient()
optimizer = optimizers.Adam(learning_rate=hyperparam["learning_rate"])

# Load model if model_path is provided, useful for continuing training
# Otherwise create a new model
model = None
if hyperparam.get("model_path"):
    model: Functional = load_model(hyperparam["model_path"], custom_objects={"DiceCoefficient": dice_coef})
else:
    model = create_unet(hyperparam["input_shape"], hyperparam["num_classes"],
                   hyperparam["initial_filters"], hyperparam["max_depth"], hyperparam["conv_per_block"],
                   hyperparam["kernel_size"], hyperparam["activation"], hyperparam["padding"])
model.compile(optimizer=optimizer, loss=losses.categorical_crossentropy, metrics=['accuracy', meaniou_metric, dice_coef])
model.summary()

# Callbacks
earlystopping = EarlyStopping(monitor="one_hot_mean_io_u", patience=10, mode="max")
checkpoint = ModelCheckpoint(filepath=hyperparam["output_name"], monitor="one_hot_mean_io_u", save_best_only=True, mode="max", verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor="one_hot_mean_io_u", factor=0.9, patience=6)
log_best_epoch = LogBestEpoch('one_hot_mean_io_u', ['accuracy', "dice_coefficient"], output_name="unet_metrics.json")


# Get training parameters from hyperparameters
initial_epoch = 0
if hyperparam.get("initial_epoch"):
    initial_epoch = hyperparam["initial_epoch"]

batch_size = 2
if hyperparam.get("batch_size"):
    initial_epoch = hyperparam["batch_size"]

max_epochs = 5000
if hyperparam.get("max_epochs"):
    initial_epoch = hyperparam["max_epochs"]

# Train Model
model.fit(x=train_images, y=train_labels, 
            epochs=max_epochs, batch_size=batch_size, callbacks=[earlystopping, checkpoint, lr_scheduler, log_best_epoch],
            shuffle=True, validation_data=(val_images, val_labels), initial_epoch=initial_epoch)
