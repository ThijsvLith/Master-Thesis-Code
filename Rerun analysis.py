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
df_rerun2 = df_rerun2.iloc[:163, :]

ic(df_main)

params = [5, 6, 110, 132]
param_names = ['Cd', 'Cl', 'Cpwu11', 'Cpwl11']

# Prepare figure with 4 subplots (vertical)
fig, axs = plt.subplots(4, 1, figsize=(10, 16), sharex=True)

dfs = [
    (df_main, 'Main'),
    (df_rerun1, 'Rerun1'),
    (df_rerun2, 'Rerun2')
]

colors = ['C0', 'C1', 'C2']

for i, (col, name) in enumerate(zip(params, param_names)):
    for df, label, color in zip([df_main, df_rerun1, df_rerun2], ['Main', 'Rerun1', 'Rerun2'], colors):
        data = df[col].reset_index(drop=True)  # Reset index so x-axis is sample number
        cummean = data.expanding().mean()
        axs[i].plot(data.index, data, label=f'{label} original', alpha=0.5, color=color)
        axs[i].plot(data.index, cummean, label=f'{label} cumulative mean', linewidth=2, linestyle='--', color=color)
    axs[i].set_ylabel(name)
    axs[i].legend()
    axs[i].grid(True)

axs[-1].set_xlabel('Row index')
plt.tight_layout()
plt.show()
