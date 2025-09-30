import pandas as pd
from icecream import ic
import matplotlib.pyplot as plt
## Multiple run analysis.

# File paths
base = r"C:\TU_Delft\Master\Thesis\Wind tunnel analysis code\Thijs LTT\Test_TvL\V3\V3_zz_0.05c_top_Re_1e6"
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
df_rerun2 = df_rerun2.iloc[:110, :]  # Make sample count equal to rerun1

# ...existing code...

ic(df_main)

params = [5, 6, 110, 132]
param_names = ['Cd', 'Cl', 'Cpwu11', 'Cpwl11']

# Prepare figure with 4 subplots (vertical)
fig, axs = plt.subplots(4, 1, figsize=(16, 16), sharex=True)

dfs = [
    (df_main, 'Main'),
    (df_rerun1, 'Rerun1'),
    (df_rerun2, 'Rerun2')
]

colors = ['C0', 'C1', 'C2']

ax_right0 = None
# Math mode labels for y-axis
math_labels = [
    r'$C_\mathrm{d}$ (-)',
    r'$C_\mathrm{l}$ (-)',
    r'$C_{p,u,11}$ (-)',
    r'$C_{p,l,11}$ (-)'
]

for i, (col, name) in enumerate(zip(params, param_names)):
    ax = axs[i]
    ax_right = ax.twinx()

    # Determine the number of samples (all runs have same length)
    num_samples = len(df_main)
    ax.set_xlim(0, num_samples - 1)  # Set x-axis to cover all measurements

    for df, label, color in zip([df_main, df_rerun1, df_rerun2], ['Main', 'Rerun1', 'Rerun2'], colors):
        data = df[col].reset_index(drop=True)
        cummean = data.expanding().mean()
        ax.plot(data.index, data, label=f'{label} original', alpha=0.5, color=color)
        ax.plot(data.index, cummean, label=f'{label} cumulative mean', linewidth=2, linestyle='--', color=color)

    # Calculate absolute percentage difference of cumulative mean for rerun1 and rerun2 with respect to main
    main_cummean = df_main[col].reset_index(drop=True).expanding().mean()
    rerun1_cummean = df_rerun1[col].reset_index(drop=True).expanding().mean()
    rerun2_cummean = df_rerun2[col].reset_index(drop=True).expanding().mean()

    min_len1 = min(len(main_cummean), len(rerun1_cummean))
    min_len2 = min(len(main_cummean), len(rerun2_cummean))

    abs_perc_diff_rerun1 = 100 * abs(rerun1_cummean[:min_len1] - main_cummean[:min_len1]) / abs(main_cummean[:min_len1])
    abs_perc_diff_rerun2 = 100 * abs(rerun2_cummean[:min_len2] - main_cummean[:min_len2]) / abs(main_cummean[:min_len2])

    ax_right.plot(
        rerun1_cummean.index[:min_len1],
        abs_perc_diff_rerun1,
        linestyle=':',
        linewidth=2,
        color='C1',
        label='Rerun1 |% diff (cummean)|'
    )
    ax_right.plot(
        rerun2_cummean.index[:min_len2],
        abs_perc_diff_rerun2,
        linestyle=':',
        linewidth=2,
        color='C2',
        label='Rerun2 |% diff (cummean)|'
    )

    ax.set_ylabel(math_labels[i])
    ax.grid(True)
    ax_right.set_ylabel('Absolute percentage difference (%)', color='gray')
    ax_right.tick_params(axis='y', labelcolor='gray')
    ax_right.set_ylim(0)  # y-axis starts at 0
    if i == 0:
        ax_right0 = ax_right

# Collect handles and labels from the first axis
handles, labels = axs[0].get_legend_handles_labels()
handles_right, labels_right = ax_right0.get_legend_handles_labels()

# Combine handles and labels for legend
all_handles = handles + handles_right
all_labels = labels + labels_right

plt.subplots_adjust(bottom=0.07)

fig.legend(
    all_handles,
    all_labels,
    loc='lower center',
    bbox_to_anchor=(0.45, 0),
    ncol=4,
    fontsize=12,
    frameon=False
)

axs[-1].set_xlabel('Measurement number (-)')

# Save the figure to the specified folder
fig.savefig(r'C:\TU_Delft\Master\Thesis\Figures overleaf\Results\Rerun analysis temp fig.png', dpi=300, bbox_inches='tight')


plt.show()
