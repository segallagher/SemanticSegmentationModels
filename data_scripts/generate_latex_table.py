import pandas as pd
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(
    prog="AnalyzeWANDBProjectMetrics",
    description="Analyzes data from wandb metrics collection. Outputs mean and std for each variable",
)

parser.add_argument("-p", "--project", dest="project")
parser.add_argument("-d", "--destination", dest="destination", default="outputs")

args = parser.parse_args()

if not args.project:
    raise Exception("You must provide --project")

path = Path(args.destination)
path.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(path / f"{args.project}-analyzed.csv")

variant_map = df.set_index("variant").to_dict(orient="index")

table_header = r"""
\begin{table*}[ht]
    \centering
    \caption{PROJECT HYPERPARAM Results}
    \label{LABEL}
    \begin{tabular}{
        @{\hspace{0.25em}}
        l@{\hspace{0.25em}}
        c@{\hspace{0.25em}}
        c@{\hspace{0.25em}}
        c@{\hspace{0.25em}}
        S[table-format=9.0]@{\hspace{0.25em}}
        c@{\hspace{0.25em}}
        c@{\hspace{0.25em}}
        c@{\hspace{0.25em}}
        S[
            table-format=4.1,
            round-mode=places,
            round-precision=1,
        ]@{\hspace{0.25em}}
    }"""

table_columns = r"""
        \hline
        {HYPERPARAM}
        & {mIoU}
        & {Accuracy}
        & {Dice}
        & {\#Parameters}
        & {\#Seconds}
        & {\#Epochs}
        & {FLOPs} 
        & {Mem(MB)} \\"""

table_footer = r"""
    \end{tabular}
\end{table*}
"""

def generate_row(
    hyperparam_value:str,
    mIoU_mean:float, mIoU_std:float,
    acc_mean:float, acc_std:float,
    dice_mean:float, dice_std:float,
    num_parameters: int,
    seconds_mean:float, seconds_std:float,
    epochs_mean:float, epochs_std:float,
    FLOPs_mean:float, FLOPs_std:float,
    memory:float,
) -> str:
    row = ""
    row += f"\n\t\t{hyperparam_value} &"
    row += f"""
        \\multicolumn{{1}}{{c}}{{
            \\num[
                table-format=1.3,
                round-mode = places,
                round-precision = 3,
            ]{{{mIoU_mean}}}
            $\\pm$
            \\num[
                table-format=1.3,
                round-mode = places,
                round-precision = 3,
            ]{{{mIoU_std}}}
        }} &"""
    row += f"""
        \\multicolumn{{1}}{{c}}{{
            \\num[
                table-format=1.3,
                round-mode = places,
                round-precision = 3,
            ]{{{acc_mean}}}
            $\\pm$
            \\num[
                table-format=1.3,
                round-mode = places,
                round-precision = 3,
            ]{{{acc_std}}}
        }} &"""
    row += f"""
        \\multicolumn{{1}}{{c}}{{
            \\num[
                table-format=1.3,
                round-mode = places,
                round-precision = 3,
            ]{{{dice_mean}}}
            $\\pm$
            \\num[
                table-format=1.3,
                round-mode = places,
                round-precision = 3,
            ]{{{dice_std}}}
        }} &"""
    row += f"\n\t\t{int(num_parameters)} &"
    row += f"""
        \\multicolumn{{1}}{{c}}{{
            \\makebox[3em][r]{{
                \\num[
                    round-mode = places,
                    round-precision = 1,
                ]{{{seconds_mean}}}
            }}
            $\\pm$
            \\makebox[3em][r]{{
                \\num[
                    round-mode = places,
                    round-precision = 1,
                ]{{{seconds_std}}}
            }}
        }} &"""
    row += f"""
        \\multicolumn{{1}}{{c}}{{
            \\makebox[2.5em][r]{{
                \\num[
                    table-format=3.1,
                    round-mode = places,
                    round-precision = 1,
                ]{{{epochs_mean}}}
            }}
            $\\pm$
            \\makebox[2.5em][r]{{
                \\num[
                    table-format=2.1,
                    round-mode = places,
                    round-precision = 1,
                ]{{{epochs_std}}}
            }}
        }} &"""
    row += f"""
        \\multicolumn{{1}}{{c}}{{
            \\num[
                exponent-mode = scientific,
                output-exponent-marker = e,
                round-mode = figures,
                round-precision = 3,
            ]{{{FLOPs_mean}}}
            $\\pm$
            \\num[
                exponent-mode = scientific,
                output-exponent-marker = e,
                round-mode = figures,
                round-precision = 2,
            ]{{{FLOPs_std}}}
        }} &"""
    row += f"""
        {memory} \\\\"""
    return row

def generate_table(
        hyperparam:str,
        project:str,
        title_param:str=None,
        variant_column:str=None,
        project_formatted_name:str=None,
    ):
    title = title_param if title_param else hyperparam.capitalize()
    table = (
        table_header
        .replace("HYPERPARAM", title, 1)
        .replace("PROJECT", project_formatted_name if project_formatted_name else project.capitalize(), 1)
        .replace("LABEL", f"tab:{args.project}-{hyperparam}-table", 1)
    )
    col_name = variant_column if variant_column else hyperparam.capitalize()
    table += table_columns.replace("HYPERPARAM", col_name, 1)

    table += "\n\t\t\\hline"
    for run in comparisons[hyperparam]:
        row = generate_row(
            run[1],
            variant_map[run[0]]["mean_mIoU"], variant_map[run[0]]["std_mIoU"], 
            variant_map[run[0]]["mean_Accuracy"], variant_map[run[0]]["std_Accuracy"], 
            variant_map[run[0]]["mean_Dice"], variant_map[run[0]]["std_Dice"],
            variant_map[run[0]]["mean_#Param"],
            variant_map[run[0]]["mean_#Seconds"], variant_map[run[0]]["std_#Seconds"], 
            variant_map[run[0]]["mean_#Epochs"], variant_map[run[0]]["std_#Epochs"], 
            variant_map[run[0]]["mean_FLOPs"], variant_map[run[0]]["std_FLOPs"], 
            variant_map[run[0]]["memory(MB)"]
        )
        table += row
    table += "\n\t\t\\hline"
    table += table_footer

    path = Path(args.destination)
    path.mkdir(parents=True, exist_ok=True)

    with open(path / f"{project}_{hyperparam}_table.txt", "w") as f:
        f.write(table)

if args.project == "autoencoder":
    comparisons = {
        "filters": [
            ("0_25xfilters", "0.25x"),
            ("0_5xfilters", "0.5x"),
            ("basis", "1x"),
            ("2xfilters", "2x"),
            ("4xfilters", "4x"),
            ("8xfilters", "8x"),
        ],
        "depth": [
            ("minus2depth", "1"),
            ("minus1depth", "2"),
            ("basis", "3"),
            ("plus1depth", "4"),
            ("plus2depth", "5"),
            ("plus3depth", "6"),
        ],
        "kernel": [
            ("1x1kernel", "1x1"),
            ("2x2kernel", "2x2"),
            ("basis", "3x3"),
            ("5x5kernel", "5x5"),
            ("7x7kernel", "7x7"),
            ("9x9kernel", "9x9"),
            ("11x11kernel", "11x11"),
        ],
        "convolutions": [
            ("minus1conv_per_block", "1"),
            ("basis", "2"),
            ("plus1conv_per_block", "3"),
            ("plus2conv_per_block", "4"),
        ],
    }

    generate_table(
        hyperparam="filters",
        project=args.project,
        title_param="\\#Filters",
    )

    generate_table(
        hyperparam="depth",
        project=args.project,
    )

    generate_table(
        hyperparam="kernel",
        project=args.project,
    )

    generate_table(
        hyperparam="convolutions",
        project=args.project,
        title_param="\\#Convolutions Per Block",
        variant_column="Conv",
    )

elif args.project == "unet":
    comparisons = {
        "filters": [
            ("0_25xinitial_filters", "0.25x"),
            ("0_5xinitial_filters", "0.5x"),
            ("basis", "1x"),
            ("2xinitial_filters", "2x"),
            ("4xinitial_filters", "4x"),
            # ("8xinitial_filters", "8x"),
        ],
        "depth": [
            # ("minus2max_depth", "1"),
            ("minus1max_depth", "1"),
            ("basis", "2"),
            ("plus1max_depth", "3"),
            ("plus2max_depth", "4"),
            ("plus3max_depth", "5"),
        ],
        "kernel": [
            ("1x1kernel", "1x1"),
            ("2x2kernel", "2x2"),
            ("basis", "3x3"),
            ("5x5kernel", "5x5"),
            ("7x7kernel", "7x7"),
            ("9x9kernel", "9x9"),
            ("11x11kernel", "11x11"),
        ],
        "convolutions": [
            ("minus1conv_per_block", "1"),
            ("basis", "2"),
            ("plus1conv_per_block", "3"),
            ("plus2conv_per_block", "4"),
        ],
    }

    generate_table(
        hyperparam="filters",
        project=args.project,
        title_param="\\#Filters",
        project_formatted_name="U-Net",
    )

    generate_table(
        hyperparam="depth",
        project=args.project,
        project_formatted_name="U-Net",
    )

    generate_table(
        hyperparam="kernel",
        project=args.project,
        project_formatted_name="U-Net",
    )

    generate_table(
        hyperparam="convolutions",
        project=args.project,
        title_param="\\#Convolutions Per Block",
        variant_column="Conv",
        project_formatted_name="U-Net",
    )
