from keras import losses, optimizers, metrics, models, callbacks, src
import tensorflow as tf
from pathlib import Path
from utils import load_data, DiceCoefficient, LogTrainingMetrics, get_param_count, get_hyperparam, get_model_summary_string, get_mem_size
from model import create_unet
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from wandb.integration.keras import WandbMetricsLogger
import os

# Initialize wandb
run = wandb.init(
   project="unet",
   name=f"{Path(__file__).parent.name}",
   group="Sweep",
)

# Get Hyperparameters
hyperparam = get_hyperparam()
print(f"Hyperparameters: {hyperparam}")

# Load Data
data_dir = Path(hyperparam["data_path"]).resolve()

if not data_dir.exists():
   raise Exception("Could not find data directory")

train_images, train_labels, val_images, val_labels, test_images, test_labels = load_data(data_dir, hyperparam["num_classes"], hyperparam["color_mapping"])

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

# Create Output Directory
output_dir = "output_dir"
if hyperparam.get("output_dir"):
    output_dir = hyperparam["output_dir"]
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Create model metrics
meaniou_metric = metrics.OneHotMeanIoU(num_classes=hyperparam["num_classes"])
dice_coef = DiceCoefficient()
config=wandb.config
lr = config.learning_rate
optimizer = optimizers.Adam(learning_rate=lr)

# Load model if model_path is provided, useful for continuing training
# Otherwise create a new model
model: src.Functional = None
if hyperparam.get("model_path"):
    model = models.load_model(hyperparam["model_path"], custom_objects={"DiceCoefficient": dice_coef})
else:
    model = create_unet(hyperparam["input_shape"], hyperparam["num_classes"],
                   hyperparam["initial_filters"], hyperparam["max_depth"], hyperparam["conv_per_block"],
                   hyperparam["kernel_size"], hyperparam["activation"], hyperparam["padding"])
model.compile(optimizer=optimizer, loss=losses.categorical_crossentropy, metrics=['accuracy', meaniou_metric, dice_coef])
model.summary()

# Get parameter counts
summary_str = get_model_summary_string(model=model)
total_param, trainable_param, non_train_param = get_param_count(summary=summary_str)
memory = get_mem_size(summary=summary_str)

# Set log directory
logdir= output_dir / "logs"

# Get training parameters from hyperparameters
initial_epoch = 0
if hyperparam.get("initial_epoch"):
    initial_epoch = hyperparam["initial_epoch"]

batch_size = 2
if getattr(config, "batch_size", None) is not None:
   batch_size = config.batch_size
elif hyperparam.get("batch_size"):
    batch_size = hyperparam["batch_size"]

max_epochs = 5000
if getattr(config, "epochs", None) is not None:
   max_epochs = config.epochs
elif hyperparam.get("max_epochs"):
    max_epochs = hyperparam["max_epochs"]

# Callbacks
earlystopping = callbacks.EarlyStopping(monitor="val_one_hot_mean_io_u", patience=10, mode="max")
checkpoint = callbacks.ModelCheckpoint(filepath=output_dir / hyperparam["output_name"], monitor="val_one_hot_mean_io_u", save_best_only=True, mode="max", verbose=1)
lr_scheduler = callbacks.ReduceLROnPlateau(monitor="val_one_hot_mean_io_u", factor=0.2, patience=5, mode='max')
log_training_metrics = LogTrainingMetrics('val_one_hot_mean_io_u', ['val_accuracy', "val_dice_coefficient"], output_path=output_dir / "training_metrics.json",
                                          total_param=total_param, trainable_param=trainable_param, non_train_param=non_train_param, memory=memory,
                                          forward_FLOPs_per_epoch=1,
                                          )
wandb_cb = WandbMetricsLogger()

# Train Model
history = model.fit(x=train_images, y=train_labels, 
            epochs=max_epochs, batch_size=batch_size, callbacks=[earlystopping, checkpoint, lr_scheduler, log_training_metrics, wandb_cb],
            shuffle=True, validation_data=(val_images, val_labels), initial_epoch=initial_epoch)

best_iou = max(history.history["val_one_hot_mean_io_u"])
wandb.log({
   "val_one_hot_mean_io_u": float(best_iou)
})

wandb.finish()
