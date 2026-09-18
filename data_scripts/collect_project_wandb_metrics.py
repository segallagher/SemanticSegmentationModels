import json
import tempfile
import pandas as pd
import wandb
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(
    prog="CollectWANDBProjectMetrics",
    description="Collects and orders multiple run metrics for a wandb project for this project",
)

parser.add_argument("-p", "--project", dest="project")
parser.add_argument("-g", "--group", dest="group", default=None)
parser.add_argument("-d", "--destination", dest="destination", default="outputs")

args = parser.parse_args()

if not args.project:
    raise Exception("You must provide --project")

# Get Config

wandb.login()

api = wandb.Api()
entity = api.default_entity

runs = api.runs(f"{entity}/{args.project}")

runs_by_variant = {}

for run in runs:
    # Skip run if archived
    if run.group != args.group:
        continue
    if run.name not in runs_by_variant:
        runs_by_variant[run.name] = [run]
    else:
        runs_by_variant[run.name].append(run)

columns = [
    "variant",
    "run_id",
    "run_number",
    "mIoU",
    "Accuracy",
    "Dice",
    "#Param",
    "#Seconds",
    "#Epochs",
    "FLOPs",
    "memory_usage",
    "GPU",
]
df = pd.DataFrame(columns=columns)

for variant in runs_by_variant:
    runs_by_variant[variant].sort(
        key=lambda variant: variant.created_at
    )
    for run_num, run in enumerate(runs_by_variant[variant]):
        # # Get hyperparameters
        # artifacts = list(run.logged_artifacts())
        # hyperparameters = {}
        # for artifact in artifacts:
        #     print(artifact.name, artifact.type, artifact.version)
        #     if f"hyperparameters-{run.id}" in artifact.name:
        #         download_dir = artifact.download(
        #             root=f"./tmp"
        #         )

        #         hyperparam_path = Path(download_dir) / "hyperparameters.json"
        #         with hyperparam_path.open("r") as f:
        #             hyperparameters = json.load(f)
        # system_metrics = ge

        
        metadata_file = next(
            (
                file for file in run.files()
                if file.name == "wandb-metadata.json"
            ),
            None,
        )
        if not metadata_file:
            print("METADATA NOT FOUND")
        with tempfile.TemporaryDirectory() as temp_dir:
            downloaded = metadata_file.download(
                root=temp_dir,
                replace=True,
            )
            metadata_path = Path(downloaded.name)
            metadata = json.loads(metadata_path.read_text())

        
        df.loc[len(df)] = {
            "variant": variant,
            "run_id": run.id,
            "run_number": run_num,
            "mIoU": run.summary.get("test_one_hot_mean_io_u"),
            "Accuracy": run.summary.get("test_accuracy"),
            "Dice": run.summary.get("test_dice_coefficient"),
            "Loss": run.summary.get("epoch/val_loss"),
            "#Param": run.config.get("total_param"),
            "#TrainableParam": run.config.get("trainable_param"),
            "#Seconds": run.summary.get("_wandb.runtime"),
            "#Epochs": run.config.get("total_epoch"),
            "best_epoch": run.config.get("best_epoch"),
            "FLOPs": run.config.get("total_forward_flops"),
            "FLOPs_per_epoch": run.config.get("forward_flops_per_epoch"),
            "memory_usage": run.config.get("memory_usage"),
            "GPU": metadata["gpu"],
        }

        # if args.project == "autoencoder":
        #     df.loc[len(df)] = {
        #         "variant": variant,
        #         "run_id": run.id,
        #         "run_number": run_num,
        #         "mIoU": run.summary.get("test_one_hot_mean_io_u"),
        #         "Accuracy": run.summary.get("test_accuracy"),
        #         "Dice": run.summary.get("test_dice_coefficient"),
        #         "Loss": run.summary.get("epoch/val_loss"),
        #         "#Param": run.config.get("total_param"),
        #         "#TrainableParam": run.config.get("trainable_param"),
        #         "#Seconds": run.summary.get("_wandb.runtime"),
        #         "#Epochs": run.config.get("total_epoch"),
        #         "best_epoch": run.config.get("best_epoch"),
        #         "FLOPs": run.config.get("total_forward_flops"),
        #         "FLOPs_per_epoch": run.config.get("forward_flops_per_epoch"),
        #         "memory_usage": run.config.get("memory_usage"),
        #         "GPU": metadata["gpu"],
        #     }
        # elif args.project == "unet":
        #     df.loc[len(df)] = {
        #         "variant": variant,
        #         "run_id": run.id,
        #         "run_number": run_num,
        #         "mIoU": run.summary.get("test_one_hot_mean_io_u"),
        #         "Accuracy": run.summary.get("test_accuracy"),
        #         "Dice": run.summary.get("test_dice_coefficient"),
        #         "Loss": run.summary.get("epoch/val_loss"),
        #         "#Param": run.config.get("total_param"),
        #         "#TrainableParam": run.config.get("trainable_param"),
        #         "#Seconds": run.summary.get("_wandb.runtime"),
        #         "#Epochs": run.config.get("total_epoch"),
        #         "best_epoch": run.config.get("best_epoch"),
        #         "FLOPs": run.config.get("total_forward_flops"),
        #         "FLOPs_per_epoch": run.config.get("forward_flops_per_epoch"),
        #         "memory_usage": run.config.get("memory_usage"),
        #         "GPU": metadata["gpu"],
        #     }
    print(variant, len(runs_by_variant[variant]), [run.created_at for run in runs_by_variant[variant]])

path = Path(args.destination)
path.mkdir(parents=True, exist_ok=True)

df.to_csv(path / f"{args.project}-metrics.csv", index=False)
