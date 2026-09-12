import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    prog="AnalyzeWANDBProjectMetrics",
    description="Analyzes data from wandb metrics collection. Outputs mean and std for each variable",
)

parser.add_argument("-p", "--project", dest="project")

args = parser.parse_args()

if not args.project:
    raise Exception("You must provide --project")

df = pd.read_csv(f"{args.project}-metrics.csv")

variants = {}

for index, run in df.iterrows():
    if run["variant"] not in variants:
        variants[run["variant"]] = [run.to_dict()]
    else:
        variants[run["variant"]].append(run.to_dict())

columns = [
    "variant",
    "mean_mIoU",
    "std_mIoU",
    "mean_Accuracy",
    "std_Accuracy",
    "mean_Dice",
    "std_Dice",
    "mean_#Param",
    "std_#Param",
    "mean_#Seconds",
    "std_#Seconds",
    "mean_#Epochs",
    "std_#Epochs",
    "mean_FLOPs",
    "std_FLOPs",
]
analyzsed_df = pd.DataFrame(columns=columns)

for var_name in variants:
    accuracies = [run["Accuracy"] for run in variants[var_name]]
    dices = [run["Dice"] for run in variants[var_name]]
    mious = [run["mIoU"] for run in variants[var_name]]
    params = [run["#Param"] for run in variants[var_name]]
    seconds = [run["#Seconds"] for run in variants[var_name]]
    epochs = [run["#Epochs"] for run in variants[var_name]]
    FLOPs = [run["FLOPs"] for run in variants[var_name]]

    acc_std = np.std(accuracies)
    dice_std = np.std(dices)
    miou_std = np.std(mious)
    param_std = np.std(params)
    second_std = np.std(seconds)
    epoch_std = np.std(epochs)
    FLOP_std = np.std(FLOPs)

    acc_mean = np.mean(accuracies)
    dice_mean = np.mean(dices)
    miou_mean = np.mean(mious)
    param_mean = np.mean(params)
    second_mean = np.mean(seconds)
    epoch_mean = np.mean(epochs)
    FLOP_mean = np.mean(FLOPs)

    analyzsed_df.loc[len(analyzsed_df)] = {
        "variant": var_name,
        "mean_mIoU": miou_mean,
        "std_mIoU": miou_std,
        "mean_Accuracy": acc_mean,
        "std_Accuracy": acc_std,
        "mean_Dice": dice_mean,
        "std_Dice": dice_std,
        "mean_#Param": param_mean,
        "std_#Param": param_std,
        "mean_#Seconds": second_mean,
        "std_#Seconds": second_std,
        "mean_#Epochs": epoch_mean,
        "std_#Epochs": epoch_std,
        "mean_FLOPs": FLOP_mean,
        "std_FLOPs": FLOP_std,
    }

analyzsed_df.to_csv(f"{args.project}-analyzed.csv")
