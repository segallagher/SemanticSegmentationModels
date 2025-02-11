import wandb
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Get args
parser = argparse.ArgumentParser(description="Download and process graphs related to a given project")
parser.add_argument(
    "--cfg", 
    type=str, 
    required=True, 
    help="Path to the config file for the project"
)

args = parser.parse_args()
# Check args
if not args.cfg:
    raise Exception("cfg file not specified")

# Load Config
with open(args.cfg, 'r') as file:
    config = json.load(file)

# Check Config
if not config['project']:
    raise Exception("Project not provided in config")
if not config['hyperparam_comparisons']:
    raise Exception("hyperparam_comparisons not provided in config")

# Get comparison array
for entry in config['hyperparam_comparisons']:
    comparisons = entry

# Get API data
api = wandb.Api()
runs = api.runs(config['project'])

# Generate Graphs

run_histories = {}
run_sys_metrics = {}
for run in runs:
    run_histories[run.name] = run.history(pandas=(True), stream="default")
    run_sys_metrics[run.name] = run.history(pandas=(True), stream="events")

def create_graph(title:str, run_dict: dict, monitor:str, x_axis:str, y_label:str, base_filename:str, project:str, y_bounds:list=None, x_label:str="Epoch"):
    # For each set of runs to compare
    for comparison in config['hyperparam_comparisons']:
        
        comparison_runs = {run: run_dict[run] for run in config['hyperparam_comparisons'][comparison]}

        # Plot each run
        largest_epoch=0
        for run_name in comparison_runs:
            
            # Drop all other columns, and drop rows with nan's
            comparison_runs[run_name] = comparison_runs[run_name][[monitor, x_axis]].dropna()
            
            # Get upper bound on graphs
            if comparison_runs[run_name][x_axis].max() > largest_epoch:
                largest_epoch = comparison_runs[run_name][x_axis].max()
            
            # Plot graph
            plt.plot(comparison_runs[run_name][x_axis],comparison_runs[run_name][monitor], label=run_name)
        

        # Write run to graph
        filepath = Path(project) / comparison
        filepath.mkdir(parents=True, exist_ok=True)
        plt.xlim(0,largest_epoch)
        if y_bounds:
            plt.ylim(*y_bounds)

        plt.title(title)
        plt.grid()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.savefig(filepath / f"{base_filename}.png")
        plt.clf()

# Evaluation Metric Graphs
create_graph(
    title="Mean IoU over Epoch",
    run_dict=run_histories,
    monitor='epoch/val_one_hot_mean_io_u',
    x_axis='epoch/epoch',
    y_label="Mean IoU",
    x_label="Epoch",
    base_filename="meaniou_epoch",
    project=config['project'],
    )
create_graph(
    title="Accuracy over Epoch",
    run_dict=run_histories,
    monitor='epoch/val_accuracy',
    x_axis='epoch/epoch',
    y_label="Accuracy",
    x_label="Epoch",
    base_filename="accuracy_epoch",
    project=config['project'],
    )
create_graph(
    title="Dice Coefficient over Epoch",
    run_dict=run_histories,
    monitor='epoch/val_dice_coefficient',
    x_axis='epoch/epoch',
    y_label="Dice Coefficient",
    x_label="Epoch",
    base_filename="dice_coefficient_epoch",
    project=config['project'],
    )

# System Metric Graphs
create_graph(
    title="Memory over Time",
    run_dict=run_sys_metrics,
    monitor='system.proc.memory.rssMB',
    x_axis='_runtime',
    y_label="Process Memory (MB)",
    x_label="Seconds",
    base_filename="memory",
    project=config['project'],
    )
create_graph(
    title="GPU Utilization over Time",
    run_dict=run_sys_metrics,
    monitor='system.gpu.0.gpu',
    x_axis='_runtime',
    y_label="GPU utilization %",
    y_bounds=[0,100],
    x_label="Seconds",
    base_filename="gpu",
    project=config['project'],
    )
create_graph(
    title="CPU Utilization over Time",
    run_dict=run_sys_metrics,
    monitor='system.cpu',
    x_axis='_runtime',
    y_label="CPU utilization %",
    y_bounds=[0,10],
    x_label="Seconds",
    base_filename="cpu",
    project=config['project'],
    )