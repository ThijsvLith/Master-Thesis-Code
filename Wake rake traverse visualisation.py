## # Wake rake traverse visualisation

import pandas as pd
import matplotlib.pyplot as plt
from plot_styling import set_plot_style

set_plot_style()

# Read the first file, skipping the first two rows
filename1 = r"C:\TU_Delft\Master\Thesis\Wind tunnel analysis code\Thijs LTT\Test_TvL\Model2\Test runs\unc_all_Traverse_no_zz_8deg.txt"
data1 = pd.read_csv(filename1, sep='\t', header=None, skiprows=2)
data1 = data1.iloc[:-1, :]  # Exclude the last row

x1 = data1.iloc[:, 9]
y1 = data1.iloc[:, 5]

# Read the second file, skipping the first two rows
filename2 = r"C:\TU_Delft\Master\Thesis\Wind tunnel analysis code\Thijs LTT\Test_TvL\Model2\Model2_no_zz_Re_1e6\unc_all_Model2_no_zz_Re_1e6.txt"
data2 = pd.read_csv(filename2, sep='\t', header=None, skiprows=2)
data2_sel = data2.iloc[2488:2622, :]
x2 = data2_sel.iloc[:, 9]
y2 = data2_sel.iloc[:, 5]

# Read the third file, skipping the first two rows
filename3 = r"C:\TU_Delft\Master\Thesis\Wind tunnel analysis code\Thijs LTT\Test_TvL\Model2\Model2_zz_0.1c_top_Re_1e6\unc_all_Model2_zz_0.1c_top_Re_1e6.txt"
data3 = pd.read_csv(filename3, sep='\t', header=None, skiprows=2)
data3_sel = data3.iloc[2485:2621, :]
x3 = data3_sel.iloc[:, 9]
y3 = data3_sel.iloc[:, 5]

x3_adjusted = x3 - 100

fig, ax = plt.subplots(figsize=(10, 4))

# Plot lines with defined color cycle
ax.plot(x1, y1, color='C0', linestyle='-', label='Full traverse no_zz')
ax.plot(x2, y2, color='C1', linestyle='-', label='no_zz')
ax.plot(x3, y3, color='C2', linestyle='-', label='zz_0.1c_top')
ax.plot(x3_adjusted, y3, color='C2', linestyle='--', label='zz_0.1c_top (shifted)')

ax.set_xlabel('z-position on wake rake (mm)')
ax.set_ylabel(r"$C_\mathrm{d}$ (-)")
ax.set_xlim(0, 400)
ax.set_xticks(range(0, 401, 50))

# Reduce the number of y-axis ticks
ax.yaxis.set_major_locator(plt.MaxNLocator(5))

ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

handles, labels = ax.get_legend_handles_labels()

fig.tight_layout()
plt.subplots_adjust(bottom=0.28)  # Add space for the legend

fig.legend(
    handles, labels,
    loc='lower center',
    bbox_to_anchor=(0.5, 0),
    ncol=len(labels),
    frameon=False
)

fig.savefig('results/WakeRake_visualization.pdf')
plt.show()