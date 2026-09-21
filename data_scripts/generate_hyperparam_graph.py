import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import re

from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

x = np.array([0.25, 0.5, 1, 5, 10, 20])
y = np.array([1.2, 1.5, 2.1, 3.8, 5.5, 8.2])

def exponential(x, a, b):
    return a * np.exp(b * x)

parameters, covariance = curve_fit(exponential, x, y)


parser = argparse.ArgumentParser(
    prog="GenerateHyperparamComparisonGraphs",
    description="Creates hyperparameter comparison graphs",
)

parser.add_argument("-p", "--project", dest="project")
parser.add_argument("-d", "--destination", dest="destination", default="outputs")

args = parser.parse_args()

if not args.project:
    raise Exception("You must provide --project")

path = Path(args.destination)
path.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(path / f"{args.project}-metrics.csv")

variants = {}

for index, run in df.iterrows():
    if run["variant"] not in variants:
        variants[run["variant"]] = [run.to_dict()]
    else:
        variants[run["variant"]].append(run.to_dict())

# Create a twin axis graph of mIoU and FLOPs against hyperparameter

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 12,
})

def generate_graph(
        comparison_list: list[tuple],
        hyperparam: str = "filters",
        x_scale = lambda x: x,
        legend_loc: str = "upper left",
        logarithmic_FLOPs_axis: bool = False,
        x_axis_label: str = None,
    ):
    x_label = []
    x = []
    y1 = []
    y2 = []
    y3 = []

    for variant in comparison_list:
        for run in variants[variant[0]]:
            x_label.append(variant[2])
            x.append(variant[1])
            y1.append(run["mIoU"])
            y2.append(run["FLOPs"])
            y3.append(run["Accuracy"])

    fig, ax1 = plt.subplots(figsize=(8,5))

    # Left y-axis
    ax1.scatter(
        x, y1,
        facecolors="none",
        edgecolors="steelblue",
        linewidths=1.5,
        s=60,
        label="mIoU"
    )

    fit1 = np.polyfit(x_scale(x), y1, 2)

    x_fit = np.geomspace(x[0], x[-1], 200)
    y1_fit = np.polyval(fit1, x_scale(x_fit))
    ax1.set_ylim(0.1, 0.5)
    ax1.plot(x_fit, y1_fit, "--", color="steelblue", label=f"mIoU Trend")

    
    # ax1.scatter(x, y3, color="purple", s=80, label="Accuracy")

    # fit3 = np.polyfit(x_scale(x), y3, 2)

    # x_fit = np.geomspace(x[0], x[-1], 200)
    # y3_fit = np.polyval(fit3, x_scale(x_fit))
    # ax1.plot(x_fit, y3_fit, "--", color="purple", label=f"Accuracy Trend")



    # Right y-axis
    ax2 = ax1.twinx()
    ax2.scatter(
        x,
        y2,
        color="darkorange",
        s=60,
        label="FLOPs (Log)" if logarithmic_FLOPs_axis else "FLOPs",
    )

    if logarithmic_FLOPs_axis:
        fit2 = np.polyfit(x_scale(x), np.log(y2), 1)
        log_y2_fit = np.polyval(fit2, x_scale(x_fit))
        y2_fit = np.exp(log_y2_fit)
        ax2.set_yscale("log")
    else:
        fit2 = np.polyfit(x_scale(x), y2, 2)
        y2_fit = np.polyval(fit2, x_scale(x_fit))



    ax2.plot(
        x_fit,
        y2_fit,
        "-",
        color="darkorange",
        label="FLOPs Log Trend" if logarithmic_FLOPs_axis else "FLOPs Trend",
    )

    # Plot functions
    equation1 = fr"--  $y={fit1[0]:.3f}x^2{fit1[1]:+.3f}x{fit1[2]:+.3f}$"
    fig.text(
        0.25, 0.1,
        equation1,
        ha="center",
        color="tab:blue",
        fontsize=15,
        fontweight="bold",
    )
    
    if logarithmic_FLOPs_axis:
        equation2 = fr"—  $y=e^{{{float(fit2[0]):.2f}\times \log_2(x) + {float(fit2[1]):.2f}}}$"
    else:
        signs = []
        a_coef, a_exp = f"{fit2[0]:.{2}e}".split("e")
        signs.append("") if float(a_coef) > 0 else signs.append("")
        b_coef, b_exp = f"{fit2[0]:.{2}e}".split("e")
        signs.append("+") if float(b_coef) > 0 else signs.append("")
        c_coef, c_exp = f"{fit2[0]:.{2}e}".split("e")
        signs.append("+") if float(c_coef) > 0 else signs.append("")
        equation2 = (
            fr"—  $y="
            fr"{{{signs[0]}}}{{{a_coef}}}\mathrm{{e}}^{{{int(a_exp)}}}x^2"
            fr"{{{signs[1]}}}{{{b_coef}}}\mathrm{{e}}^{{{int(b_exp)}}}x"
            fr"{{{signs[2]}}}{{{c_coef}}}\mathrm{{e}}^{{{int(c_exp)}}}$"
        )
    fig.text(
        0.75, 0.1,
        s=equation2,
        ha="center",
        color="tab:orange",
        fontsize=15,
        fontweight="bold",
    )


    # Base-2 logarithmic x-axis
    if x_scale == np.log2:
        ax1.set_xscale("log", base=2)

    ax1.set_xticks(x)
    ax1.set_xticklabels(x_label)

    ax1.set_xlabel(x_axis_label)
    ax1.set_ylabel("mIoU", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2.set_ylabel(
        "FLOPs (Log)" if logarithmic_FLOPs_axis else "FLOPs",
        color="darkorange"
    )
    ax2.tick_params(axis="y", labelcolor="darkorange")

    ax1.set_title(f"mIoU and FLOPs over {hyperparam}")
    ax1.tick_params(axis="x", rotation=0)

    # Combine legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(handles1+handles2, labels1+labels2, loc=legend_loc)

    fig.tight_layout(rect=[0,.10,1,1])
    plt.savefig(path / f"{args.project}-{hyperparam.lower().replace(" ", "-")}-graph.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

if args.project == "autoencoder":

    # Filters
    filters_comparison_variants = [
        ("0_25xfilters", 0.25, "0.25x"),
        ("0_5xfilters", 0.5, "0.5x"),
        ("basis", 1, "1x"),
        ("2xfilters", 2, "2x"),
        ("4xfilters", 4, "4x"),
        ("8xfilters", 8, "8x"),
    ]
    generate_graph(
        filters_comparison_variants,
        "Num Filters",
        x_scale=np.log2,
        logarithmic_FLOPs_axis=True,
        legend_loc="lower right",
        x_axis_label = "Filters",
    )

    # Depth
    depth_comparison_variants = [
        ("minus2depth", 1, "1"),
        ("minus1depth", 2, "2"),
        ("basis", 3, "3"),
        ("plus1depth", 4, "4"),
        ("plus2depth", 5, "5"),
        ("plus3depth", 6, "6"),
    ]
    generate_graph(
        depth_comparison_variants,
        "Depth",
        legend_loc="center left",
        x_axis_label = "Depth",
    )

    # Convolutions
    convolution_comparison_variants = [
        ("minus1conv_per_block", 1, "1"),
        ("basis", 2, "2"),
        ("plus1conv_per_block", 3, "3"),
        ("plus2conv_per_block", 4, "4"),
    ]
    generate_graph(
        convolution_comparison_variants,
        "Convolutions per Block",
        legend_loc="lower right",
        x_axis_label = "Convolutions per Block",
    )

    # Kernel
    kernel_comparison_variants = [
        ("1x1kernel", 1, "1x1"),
        ("2x2kernel", 2, "2x2"),
        ("basis", 3, "3x3"),
        ("5x5kernel", 5, "5x5"),
        ("7x7kernel", 7, "7x7"),
        ("9x9kernel", 9, "9x9"),
        ("11x11kernel", 11, "11x11"),
    ]
    generate_graph(
        kernel_comparison_variants,
        "Kernel",
        legend_loc="lower right",
        x_axis_label = "Kernel",
    )
elif args.project == "unet":

    # Filters
    filters_comparison_variants = [
        ("0_25xinitial_filters", 0.25, "0.25x"),
        ("0_5xinitial_filters", 0.5, "0.5x"),
        ("basis", 1, "1x"),
        ("2xinitial_filters", 2, "2x"),
        ("4xinitial_filters", 4, "4x"),
        # ("8xfilters", 8, "8x"),
    ]
    generate_graph(
        filters_comparison_variants,
        "Num Filters",
        legend_loc="lower right",
        x_scale=np.log2,
        logarithmic_FLOPs_axis=True,
        x_axis_label = "Filters",
    )

    # Depth
    depth_comparison_variants = [
        # ("minus2max_depth", 1, "1"),
        ("minus1max_depth", 1, "1"),
        ("basis", 2, "2"),
        ("plus1max_depth", 3, "3"),
        ("plus2max_depth", 4, "4"),
        ("plus3max_depth", 5, "5"),
    ]
    generate_graph(
        depth_comparison_variants,
        "Depth",
        legend_loc="lower right",
        x_axis_label = "Depth",
    )

    # Convolutions
    convolution_comparison_variants = [
        ("minus1conv_per_block", 1, "1"),
        ("basis", 2, "2"),
        ("plus1conv_per_block", 3, "3"),
        ("plus2conv_per_block", 4, "4"),
    ]
    generate_graph(
        convolution_comparison_variants,
        "Convolutions per Block",
        legend_loc="lower right",
        x_axis_label = "Convolutions per Block",
    )

    # Kernel
    kernel_comparison_variants = [
        ("1x1kernel", 1, "1x1"),
        ("2x2kernel", 2, "2x2"),
        ("basis", 3, "3x3"),
        ("5x5kernel", 5, "5x5"),
        ("7x7kernel", 7, "7x7"),
        ("9x9kernel", 9, "9x9"),
        ("11x11kernel", 11, "11x11"),
    ]
    generate_graph(
        kernel_comparison_variants,
        "Kernel",
        legend_loc="lower right",
        x_axis_label = "Kernel",
    )