import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Union
import os
from icecream import ic
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline

from plot_styling import set_plot_style
set_plot_style()

PROCESSED_DATA_DIR = Path("processed_data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

LABEL_MAP = {
    'V3_no_zz_Re_1e6': r'No ZZ, $\mathrm{Re}=10^6$',
    'V3_small_zz_bottom_Re_1e6': r'ZZ on bottom at $90^\circ$, $\mathrm{Re}=10^6$',
    'V3_zz_0.05c_top_Re_1e6': r'ZZ on top at 5\%, $\mathrm{Re}=10^6$',
    'V3_zz_bottom_0.05c_top_Re_1e6': r'ZZ on bottom at $90^\circ$ and on top at 5\%, $\mathrm{Re}=10^6$',
    'V3_bottom_0.03c_top_Re_1e6': r'ZZ on bottom at $90^\circ$ and on top at 3\%, $\mathrm{Re}=10^6$',
    'V3_bottom_45deg_0.03c_top_Re_1e6': r'ZZ on bottom at $45^\circ$ and on top at 3\%, $\mathrm{Re}=10^6$',
    'V3_bottom_45_deg_Re_1e6': r'ZZ on bottom at $45^\circ$, $\mathrm{Re}=10^6$',

    'V3_no_zz_Re_5e5': r'No ZZ, $\mathrm{Re}=5 \times 10^5$',
    'V3_small_zz_bottom_Re_5e5': r'ZZ on bottom at $90^\circ$, $\mathrm{Re}=5 \times 10^5$',
    'V3_zz_0.05c_top_Re_5e5': r'ZZ on top at 5\%, $\mathrm{Re}=5 \times 10^5$',
    'V3_zz_bottom_0.05c_top_Re_5e5': r'ZZ on bottom at $90^\circ$ and on top at 5\%, $\mathrm{Re}=5 \times 10^5$',
    'V3_bottom_45_deg_Re_5e5': r'ZZ on bottom at $45^\circ$, $\mathrm{Re}=5 \times 10^5$',

    'Model2_no_zz_Re_1e6': r'No ZZ, $\mathrm{Re}=10^6$',
    'Model2_small_zz_bottom_Re_1e6': r'ZZ on bottom at $90^\circ$, $\mathrm{Re}=10^6$',
    'Model2_zz_0.1c_top_Re_1e6': r'ZZ on top at 10\%, $\mathrm{Re}=10^6$',
    'Model2_zz_0.05c_top_Re_1e6': r'ZZ on top at 5\%, $\mathrm{Re}=10^6$',
    'Model2_zz_bottom_0.05c_top_Re_1e6': r'ZZ on bottom at $90^\circ$ and on top at 5\%, $\mathrm{Re}=10^6$',

    'Model2_no_zz_Re_5e5': r'No ZZ, $\mathrm{Re}=5 \times 10^5$',
    'Model2_small_zz_bottom_Re_5e5': r'ZZ on bottom at $90^\circ$, $\mathrm{Re}=5 \times 10^5$',
    'Model2_zz_0.1c_top_Re_5e5': r'ZZ on top at 10\%, $\mathrm{Re}=5 \times 10^5$',
    'Model2_zz_0.05c_top_Re_5e5': r'ZZ on top at 5\%, $\mathrm{Re}=5 \times 10^5$',
    'Model2_zz_bottom_0.05c_top_Re_5e5': r'ZZ on bottom at $90^\circ$ and on top at 5\%, $\mathrm{Re}=5 \times 10^5$',
}

def get_label(Reynolds_on, casename: str) -> str:
    """Return display label for casename; fallback to casename if not in LABEL_MAP."""
    label = LABEL_MAP.get(casename, casename)
    if Reynolds_on == False:
        label = label.replace(r', $\mathrm{Re}=10^6$', '')
        label = label.replace(r', $\mathrm{Re}=5 \times 10^5$', '')
    return label

def load_processed_case(casename: str) -> pd.DataFrame:
    input_path = PROCESSED_DATA_DIR / f"{casename}_processed.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing processed data for case '{casename}'. Expected at {input_path}.")
    return pd.read_csv(input_path)

def plot_multi_case(Reynolds_on, casenames, output_path):
    fig, axs = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=False)
    ax_cl_alpha, ax_cd_alpha = axs[0, 0], axs[0, 1]
    ax_cl_cd, ax_clcd_alpha = axs[1, 0], axs[1, 1]

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx, casename in enumerate(casenames):
        df = load_processed_case(casename)
        color = colors[idx % len(colors)]
        label = get_label(Reynolds_on, casename)

        # Get indices for measured region
        start_idx = int(df['start_idx'].iloc[0])
        last_measured_idx = int(df['last_measured_idx'].iloc[0])

        # Cl-alpha: plot measured region as solid, rest as dashed, 'X' at last_measured_idx
        alpha = df["alpha"]
        Cl = df["Cl"]

        # Measured region
        ax_cl_alpha.plot(
            alpha[start_idx:last_measured_idx+1],
            Cl[start_idx:last_measured_idx+1],
            '-', color=color, label=label
        )
        # 'X' at last measured point
        ax_cl_alpha.plot(
            alpha[last_measured_idx],
            Cl[last_measured_idx],
            'x', color=color, markersize=10, markeredgewidth=2
        )
        # Unmeasured region after last_measured_idx
        if last_measured_idx < len(df):
            ax_cl_alpha.plot(
                alpha[last_measured_idx:],
                Cl[last_measured_idx:],
                '--', color=color
            )
        # Unmeasured region before start_idx (if any)
        if start_idx > 0:
            ax_cl_alpha.plot(
                alpha[:start_idx],
                Cl[:start_idx],
                '--', color=color
            )

        # Cd-alpha (only measured region)
        Cd = df["Cd"]
        ax_cd_alpha.plot(
            alpha[start_idx:last_measured_idx],
            Cd[start_idx:last_measured_idx],
            '-', color=color, label=label
        )

        # Cl-Cd (only measured region)
        ax_cl_cd.plot(
            Cd[start_idx:last_measured_idx],
            Cl[start_idx:last_measured_idx],
            '-', color=color, label=label
        )

        # Cl/Cd-alpha (only measured region)
        ax_clcd_alpha.plot(
            alpha[start_idx:last_measured_idx],
            (Cl/Cd)[start_idx:last_measured_idx],
            '-', color=color, label=label
        )

    # Axis labels
    ax_cl_alpha.set_xlabel(r"$\alpha$ ($^\circ$)")
    ax_cl_alpha.set_ylabel(r"$C_{\mathrm{l}}$ (-)")
    ax_cd_alpha.set_xlabel(r"$\alpha$ ($^\circ$)")
    ax_cd_alpha.set_ylabel(r"$C_{\mathrm{d}}$ (-)")
    ax_cl_cd.set_xlabel(r"$C_{\mathrm{d}}$ (-)")
    ax_cl_cd.set_ylabel(r"$C_{\mathrm{l}}$ (-)")
    ax_clcd_alpha.set_xlabel(r"$\alpha$ ($^\circ$)")
    ax_clcd_alpha.set_ylabel(r"$C_{\mathrm{l}}/C_{\mathrm{d}}$ (-)")

    # Axis limits and ticks (adjust as needed)
    for ax in [ax_cl_alpha, ax_cd_alpha, ax_clcd_alpha]:
        ax.set_xlim(-10, 25)
        ax.set_xticks(np.arange(-10, 26, 5))
    ax_cl_alpha.set_ylim(-0.5, 2.0)
    ax_cl_alpha.set_yticks(np.arange(-0.5, 2.6, 0.5))
    ax_cd_alpha.set_ylim(0, 0.4)
    ax_cd_alpha.set_yticks(np.arange(0, 0.41, 0.1))
    ax_clcd_alpha.set_ylim(-10, 100)
    ax_clcd_alpha.set_yticks(np.arange(0, 101, 20))

    ax_cl_cd.set_xlim(0, 0.4)
    ax_cl_cd.set_xticks(np.arange(0, 0.41, 0.1))  # Set x-axis ticks for Cd

    ax_cl_cd.set_ylim(-0.5, 2.5)         # Set y-axis limits for Cl
    ax_cl_cd.set_yticks(np.arange(-0.5, 2.6, 0.5)) # Set y-axis ticks for Cl

    # Grid
    for ax in axs.flat:
        ax.grid(True)

    # Legend at the bottom
    handles, labels = ax_cl_alpha.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5,0),
        ncol=2, #len(labels),
        frameon=False,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(output_path)
    plt.show()
    # plt.close(fig)

# CASE_GROUPS containing only the 5-casenames lists (from the original file)
CASE_GROUPS = {
    "V3_all_5e5": {
        "cases": [
            "V3_no_zz_Re_5e5",
            "V3_small_zz_bottom_Re_5e5",
            "V3_zz_0.05c_top_Re_5e5",
            "V3_zz_bottom_0.05c_top_Re_5e5",
            "V3_bottom_45_deg_Re_5e5",
        ],
        "output": RESULTS_DIR / "multi_case_comparison_V3_Re5e5.pdf",
    },
    "V3_all_1e6": {
        "cases": [
            "V3_no_zz_Re_1e6",
            "V3_small_zz_bottom_Re_1e6",
            "V3_zz_0.05c_top_Re_1e6",
            "V3_zz_bottom_0.05c_top_Re_1e6",
            "V3_bottom_0.03c_top_Re_1e6",
            "V3_bottom_45deg_0.03c_top_Re_1e6",
            "V3_bottom_45_deg_Re_1e6",
        ],
        "output": RESULTS_DIR / "multi_case_comparison_V3_Re1e6.pdf",
    },
    "Model2_all_1e6": {
        "cases": [
            "Model2_no_zz_Re_1e6",
            "Model2_small_zz_bottom_Re_1e6",
            "Model2_zz_0.1c_top_Re_1e6",
            "Model2_zz_0.05c_top_Re_1e6",
            "Model2_zz_bottom_0.05c_top_Re_1e6",
        ],
        "output": RESULTS_DIR / "multi_case_comparison_Model2_Re1e6.pdf",
    },
    "Model2_all_5e5": {
        "cases": [
            "Model2_no_zz_Re_5e5",
            "Model2_small_zz_bottom_Re_5e5",
            "Model2_zz_0.1c_top_Re_5e5",
            "Model2_zz_0.05c_top_Re_5e5",
            "Model2_zz_bottom_0.05c_top_Re_5e5",
        ],
        "output": RESULTS_DIR / "multi_case_comparison_Model2_Re5e5.pdf",
    },
    "V3_multiple_Re": {
        "cases": [
            "V3_no_zz_Re_5e5",
            "V3_no_zz_Re_1e6",
            "V3_zz_bottom_0.05c_top_Re_5e5",
            "V3_zz_bottom_0.05c_top_Re_1e6",
        ],
        "output": RESULTS_DIR / "multi_case_comparison_V3_multiple_Re.pdf",
    },
    "Model2_multiple_Re": {
        "cases": [
            "Model2_no_zz_Re_5e5",
            "Model2_no_zz_Re_1e6",
            "Model2_zz_bottom_0.05c_top_Re_5e5",
            "Model2_zz_bottom_0.05c_top_Re_1e6",
        ],
        "output": RESULTS_DIR / "multi_case_comparison_Model2_multiple_Re.pdf",
    },
}

def main():
    # Choose which group to plot by changing GROUP_KEY to one of:
    #   "V3_all_5e5", "V3_all_1e6", "Model2_all_1e6", "Model2_all_5e5", "V3_multiple_Re", "Model2_multiple_Re"
    GROUP_KEY = "Model2_multiple_Re"
    Reynolds_on = True  # Set to False to remove Re from labels

    if GROUP_KEY not in CASE_GROUPS:
        raise KeyError(f"Unknown CASE_GROUP key: {GROUP_KEY}. Available keys: {list(CASE_GROUPS.keys())}")

    casenames = CASE_GROUPS[GROUP_KEY]["cases"]
    output_path = CASE_GROUPS[GROUP_KEY]["output"]

    plot_multi_case(Reynolds_on, casenames, output_path)

if __name__ == "__main__":
    main()