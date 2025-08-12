## # Wake rake traverse visualisation

import pandas as pd
import matplotlib.pyplot as plt

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

x3_adjusted = x3-100

plt.figure(figsize=(9, 5))

# Plot lines with defined color cycle
plt.plot(x1, y1, color='C0', linestyle='-', label='Model2 Full traverse')  # blue
plt.plot(x2, y2, color='C1', linestyle='-', label='Model2 (no zigzag, Re=1e6)')  # orange
plt.plot(x3, y3, color='C2', linestyle='-', label='Model2 (zz @ 0.1c top, Re=1e6)')
plt.plot(x3_adjusted, y3, color='C2', linestyle='--', label='Model2 (shifted, zz @ 0.1c top)')

# Labels and title
plt.xlabel('z-position on wake rake [mm]', fontsize=12)
plt.ylabel('$C_d$ [-]', fontsize=12)
plt.title('Wake Rake Traverse Comparison $\\alpha = 8^\\circ$ and $Re = 1e6$', fontsize=13)

# Grid and legend
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()