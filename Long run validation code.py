import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker          #  ← NEW

# -------------------- configuration --------------------
PARAM = {5: 'Cd', 6: 'Cl', 110: 'Cpwu11', 132: 'Cpwl11'}
COLS  = [5, 6, 110, 132]

filename_short = r"C:\TU_Delft\Master\Thesis\Wind tunnel analysis code\Thijs LTT\Test_TvL\V3\V3_small_zz_bottom_Re_1e6\unc_all_V3_small_zz_bottom_Re_1e6.txt"
filename_long  = r"C:\TU_Delft\Master\Thesis\Wind tunnel analysis code\Thijs LTT\Test_TvL\V3\LongRun_measure_validation\unc_all_V3_bottom_zz_LongRun_AoA5_Re1e6.txt"

# -------------------- load data --------------------
usecols = range(145)
df_short = pd.read_csv(filename_short, sep='\t', header=None, skiprows=2, usecols=usecols)
df_long  = pd.read_csv(filename_long,  sep='\t', header=None, skiprows=2, usecols=usecols)

# Select rows 1678–1788 (Excel indexing → subtract 1 for pandas)
df_short = df_short.iloc[1675:1786, :]

# -------------------- plot --------------------
# fig, axs = plt.subplots(
#     nrows=4, ncols=1,
#     figsize=(10, 14),
#     sharex=True
# )

# Math mode labels for y-axis
math_labels = [
    r'$C_\mathrm{d}$ (-)',
    r'$C_\mathrm{l}$ (-)',
    r'$C_{p,u,11}$ (-)',
    r'$C_{p,l,11}$ (-)'
]

fig, axs = plt.subplots(4, 1, figsize=(16, 16), sharex=True)

for ax, col, math_label in zip(axs, COLS, math_labels):
    y_short = df_short.iloc[:, col].reset_index(drop=True)
    y_long  = df_long.iloc[:,  col].reset_index(drop=True)

    # Cumulative means
    short_mean = y_short.expanding(min_periods=1).mean()
    long_mean  = y_long.expanding(min_periods=1).mean()

    # Make x-axis as long as the long run measurements, starting from zero
    x_long = range(len(y_long))
    x_short = range(len(y_short))

    # Raw data
    ax.plot(x_short, y_short, label='short raw', color='C0', alpha=0.5)
    ax.plot(x_short, short_mean, label='short cumulative mean', color='C0', linewidth=2, linestyle='--', alpha=1)

    ax.plot(x_long,  y_long,  label='long raw',  color='C1', alpha=0.5)
    ax.plot(x_long,  long_mean,  label='long cumulative mean',  color='C1', linewidth=2,linestyle='--', alpha=1)
    # Cumulative means
    
    

    # Y-axis padding (10 %)
    y_min, y_max = min(y_short.min(), y_long.min()), max(y_short.max(), y_long.max())
    pad = 0.10 * (y_max - y_min)
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.yaxis.set_major_locator(mticker.MaxNLocator(6))
    ax.set_ylabel(math_label)
    ax.grid(True)

    # Set x-axis limits to match long run measurements
    ax.set_xlim(0, len(y_long) - 1)

    if ax is not axs[-1]:
        ax.set_xlabel('')
        ax.label_outer()

# Set x-axis label for the last subplot
axs[-1].set_xlabel('Measurement number (-)')

plt.subplots_adjust(bottom=0.07)

handles, labels = axs[0].get_legend_handles_labels()

# Place a single legend at the bottom center of the figure
fig.legend(
    handles,
    labels,
    loc='lower center',
    bbox_to_anchor=(0.45, 0),
    ncol=len(labels),
    fontsize=12,
    frameon=False
)

# plt.tight_layout()
plt.savefig(r"C:\TU_Delft\Master\Thesis\Figures overleaf\Results\Long_run_validation_new.png", dpi=300, bbox_inches='tight')
plt.show()