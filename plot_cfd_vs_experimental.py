"""Create comparison plots for CFD and experimental datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_styling import set_plot_style
from utils import PROCESSED_DATA_DIR, PROJECT_DIR, RESULTS_DIR, ensure_directory

# DEFAULT_CASES = ["V3_no_zz_Re_1e6"]
# DEFAULT_CASES = ["Model2_no_zz_Re_1e6"]
# DEFAULT_CASES = ["V3_no_zz_Re_5e5"]
# DEFAULT_CASES = ["Model2_no_zz_Re_5e5"]
DEFAULT_CASES = ['V3_bottom_45deg_0.03c_top_Re_1e6']

def parse_case_info(case_string: str) -> Tuple[str, str]:
    parts = case_string.split("_")
    model = parts[0]
    Re = None
    for part in parts:
        if part.startswith("Re"):
            Re = part.split("Re")[1]
            if not Re:
                idx = parts.index(part)
                Re = parts[idx + 1]
            break
    if Re is None:
        raise ValueError(f"Could not parse Reynolds number from case '{case_string}'.")
    return model, Re


def data_cfd(model: str, Re: str, transition: bool) -> pd.DataFrame:
    root_path = PROJECT_DIR / "CFD results"
    model_folder = model if model in {"V3", "Model2"} else model.capitalize()
    model_path = root_path / model_folder
    if not model_path.exists():
        raise FileNotFoundError(f"Missing CFD model folder: {model_path}")

    if Re == "1e6":
        data_file = "Re1e6 transition" if transition else "Re1e6 fully turbulent"
    elif Re == "5e5":
        data_file = "Re5e5 transition" if transition else "Re5e5 fully turbulent"
    else:
        raise ValueError("Re must be '1e6' or '5e5'.")

    file_path = model_path / data_file / "polar.dat"
    if not file_path.exists():
        raise FileNotFoundError(f"Could not locate CFD data '{file_path}'.")

    with file_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split()
    df = pd.read_csv(file_path, sep=r"\s+", header=None, skiprows=1, names=header)
    if "iter_n" in df.columns:
        df = df[df["iter_n"] != 5001]
    return df


def load_processed_case(casename: str) -> pd.DataFrame:
    input_path = PROCESSED_DATA_DIR / f"{casename}_processed.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing processed data for case '{casename}'. Expected at {input_path}."
        )
    df = pd.read_csv(input_path)
    bool_cols = [
        "is_measured",
        "include_measured_plot",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    return df


def plot_case(df: pd.DataFrame, casename: str, output_path: Path) -> None:
    set_plot_style()
    model, Re = parse_case_info(casename)
    transition_value = df["transition"].iloc[0] if "transition" in df else True
    if isinstance(transition_value, str):
        transition = transition_value.strip().lower() == "true"
    else:
        transition = bool(transition_value)

    cfd_results = data_cfd(model, Re, transition)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=False)
    for ax in (ax1, ax2):
        ax.grid(True)
        ax.tick_params()

    measured_mask = (
        df["include_measured_plot"]
        if "include_measured_plot" in df
        else pd.Series(True, index=df.index)
    )
    measured_df = df[measured_mask]
    if not measured_df.empty:
        ax1.plot(
            measured_df["alpha"],
            measured_df["Cl"],
            "-x",
            linewidth=2,
            color="C0",
            label="WT",
        )
        ax2.plot(
            measured_df["alpha"],
            measured_df["Cd"],
            "-x",
            linewidth=2,
            color="C0",
            label="WT",
        )

    if not measured_df.empty and {"std_Cl", "std_Cd"}.issubset(measured_df.columns):
        ax1.fill_between(
            measured_df["alpha"],
            measured_df["Cl"] - 3 * measured_df["std_Cl"],
            measured_df["Cl"] + 3 * measured_df["std_Cl"],
            color="C0",
            alpha=0.3,
            label="WT CI of 99\%",
        )
        # print(measured_df["std_Cl"])
        ax2.fill_between(
            measured_df["alpha"],
            measured_df["Cd"] - 3 * measured_df["std_Cd"],
            measured_df["Cd"] + 3 * measured_df["std_Cd"],
            color="C0",
            alpha=0.3,
            label="_nolegend_",
        )

    region_series = (
        df["region"] if "region" in df else pd.Series("measured", index=df.index)
    )
    extrap_df = df[region_series == "extrapolated"]
    if not extrap_df.empty:
        ax1.plot(
            extrap_df["alpha"],
            extrap_df["Cl_mid"],
            "-x",
            linewidth=2,
            color="red",
            label=r"Extrapolated $C_{\mathrm{l}}$",
        )
        if {"Cl_low", "Cl_up"}.issubset(extrap_df.columns):
            ax1.fill_between(
                extrap_df["alpha"],
                extrap_df["Cl_low"],
                extrap_df["Cl_up"],
                color="red",
                alpha=0.3,
                label="Extrapolated uncertainty",
            )

    ax1.plot(
        cfd_results["Alpha"],
        cfd_results["Cl"],
        "--o",
        linewidth=2,
        color="g",
        label="CFD",
    )
    ax1.set_xlabel(r"$\alpha$ ($^\circ$)")
    ax1.set_ylabel(r"$C_{\mathrm{l}}$ (-)")

    ax2.plot(
        cfd_results["Alpha"],
        cfd_results["Cd"],
        "--o",
        linewidth=2,
        color="g",
        label="CFD",
    )
    ax2.set_xlabel(r"$\alpha$ ($^\circ$)")
    ax2.set_ylabel(r"$C_{\mathrm{d}}$ (-)")

    # Automatically set axis limits and ticks based on data ranges with some padding
    # def set_axis_limits_and_ticks(
    #     ax,
    #     xdata,
    #     ydata,
    #     xpad=1,
    # ):
    #     xmin, xmax = np.min(xdata), np.max(xdata)
    #     ymin, ymax = np.min(ydata), np.max(ydata)
    #     ax.set_xlim(xmin - xpad, xmax + xpad)
    #     # ax.set_ylim(ymin * 1.01, ymax * 1.01)

    # # Set limits for ax1 (Cl vs alpha)
    # set_axis_limits_and_ticks(
    #     ax1,
    #     np.concatenate([df["alpha"].values, cfd_results["Alpha"].values]),
    #     np.concatenate([df["Cl"].values, cfd_results["Cl"].values]),
    # )

    # # Set limits for ax2 (Cd vs alpha)
    # set_axis_limits_and_ticks(
    #     ax2,
    #     np.concatenate([df["alpha"].values, cfd_results["Alpha"].values]),
    #     np.concatenate([df["Cd"].values, cfd_results["Cd"].values]),
    # )

    for ax in [ax1, ax2]:
        ax.set_xlim(-10, 25)
        ax.set_xticks(np.arange(-10, 26, 5))

# Example: Set y-axis limits (adjust as needed for your data)
    ax1.set_ylim(-0.5, 2.0)   # Cl axis
    ax1.set_yticks(np.arange(-0.5, 2.6, 0.5))

    ax2.set_ylim(0, 0.4)      # Cd axis
    ax2.set_yticks(np.arange(0, 0.41, 0.1))

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
    )

    ensure_directory(RESULTS_DIR)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    fig.savefig(output_path)
    # plt.show()
    plt.close(fig)


def plot_cases(cases: Iterable[str]) -> List[Tuple[str, Path]]:
    outputs: List[Tuple[str, Path]] = []
    ensure_directory(RESULTS_DIR)
    for casename in cases:
        df = load_processed_case(casename)
        output_path = RESULTS_DIR / f"{casename}_comparison.pdf"
        plot_case(df, casename, output_path)
        outputs.append((casename, output_path))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CFD vs experimental data using preprocessed CSV inputs.",
    )
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        help="Case identifier to plot (can be provided multiple times).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = args.cases if args.cases else DEFAULT_CASES
    plot_cases(cases)


if __name__ == "__main__":
    main()
