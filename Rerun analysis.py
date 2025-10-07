import pandas as pd
import matplotlib.pyplot as plt
from plot_styling import set_plot_style
import numpy as np
from icecream import ic

set_plot_style()

# File paths
base = r"V3\V3_zz_0.05c_top_Re_1e6"
file_main = base + r"\unc_all_V3_zz_0.05c_top_Re_1e6.txt"
file_rerun1 = base + r"\Stall_Rerun1\unc_all_V3_zz_0.05c_top_Re_1e6_Stall_Rerun.txt"
file_rerun2 = base + r"\Stall_Rerun2\unc_all_V3_zz_0.05c_top_Re_1e6_rerun2.txt"

# Read main file, skip first two rows, select Excel rows 2233–2343 (pandas is 0-based, so 2232:2343)
df_main = pd.read_csv(file_main, sep='\t', header=None, skiprows=2, usecols=range(145))
df_main = df_main.iloc[2232:2342, :]

# Read rerun files, skip first two rows, select Excel rows 3–166 (pandas is 0-based, so 2:166)
df_rerun1 = pd.read_csv(file_rerun1, sep='\t', header=None, skiprows=2, usecols=range(145))
df_rerun1 = df_rerun1.iloc[:110, :]

df_rerun2 = pd.read_csv(file_rerun2, sep='\t', header=None, skiprows=2, usecols=range(145))
# df_rerun2 = df_rerun2.iloc[:110, :]  # Make sample count equal to rerun1
df_rerun2 = df_rerun2.iloc[:163, :]  # For all measurements

# ic(df_rerun2)

params = [5, 6, 110, 132]
param_names = ['Cd', 'Cl', 'Cpwu11', 'Cpwl11']

# Math mode labels for y-axis (with primes for Cd and Cl)
math_labels = [
    r"$C_\mathrm{d}'$ (-)",
    r"$C_\mathrm{l}'$ (-)",
    r'$C_\mathrm{p,u,11}$ (-)',
    r'$C_\mathrm{p,l,11}$ (-)'
]

fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

colors = ['C0', 'C1', 'C2']
ax_right0 = None

for i, (col, name) in enumerate(zip(params, param_names)):
    ax = axs[i]

    # Determine the number of samples (all runs have same length)
    num_samples = len(df_rerun2)
    ax.set_xlim(0, num_samples - 1)  # Set x-axis to cover all measurements

    for df, label, color in zip([df_main, df_rerun1, df_rerun2], ['Main', 'Rerun1', 'Rerun2'], colors):
        data = df[col].reset_index(drop=True)
        cummean = data.expanding().mean()
        ax.plot(data.index, data, label=f'{label} raw', alpha=0.5, color=color)
        ax.plot(data.index, cummean, label=f'{label} cumulative mean', linewidth=2, linestyle='--', color=color)

    # --- calculate final absolute percentage differences (single values) ---
    main_cummean   = df_main[col].reset_index(drop=True).expanding().mean()
    rerun1_cummean = df_rerun1[col].reset_index(drop=True).expanding().mean()
    rerun2_cummean = df_rerun2[col].reset_index(drop=True).expanding().mean()

    # Use the final value (last element) of each cumulative mean for comparison
    if len(main_cummean) == 0:
        perc1 = np.nan
        perc2 = np.nan
    else:
        main_final = main_cummean.iloc[-1]

        # Rerun1 final percentage difference w.r.t. main final cumulative mean
        if len(rerun1_cummean) == 0 or abs(main_final) < 1e-12:
            perc1 = np.nan
        else:
            rerun1_final = rerun1_cummean.iloc[-1]
            perc1 = 100.0 * abs(rerun1_final - main_final) / abs(main_final)

        # Rerun2 final percentage difference w.r.t. main final cumulative mean
        if len(rerun2_cummean) == 0 or abs(main_final) < 1e-12:
            perc2 = np.nan
        else:
            rerun2_final = rerun2_cummean.iloc[-1]
            perc2 = 100.0 * abs(rerun2_final - main_final) / abs(main_final)

    # Print results to terminal
    print(f"{name}: Rerun1 final abs % diff = {perc1:.2f}% ; Rerun2 final abs % diff = {perc2:.2f}%")

    # set y-label for left axis only (no right axis)
    ax.set_ylabel(math_labels[i])
    ax.grid(True)

# Collect handles and labels from the first axis only (no right-axis handles)
handles, labels = axs[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.01),
    ncol=3,
    frameon=False
)

axs[-1].set_xlabel('Measurement number (-)')

fig.tight_layout()
plt.subplots_adjust(bottom=0.15)
# Save the figure to the specified folder as PDF
# fig.savefig('results/rerun_analysis.pdf')

plt.show()
