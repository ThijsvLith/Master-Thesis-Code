import numpy as np
import matplotlib.pyplot as plt

# File paths for AoA2
# cf_file = r"C:\TU_Delft\Master\Thesis\Kasper results models\Thijs results\V3\Re1e6 fully turbulent\AOA_2\postProcessing\surfaces\1045\cf_AOA_2.dat"
# cp_file = r"C:\TU_Delft\Master\Thesis\Kasper results models\Thijs results\V3\Re1e6 fully turbulent\AOA_2\postProcessing\surfaces\1045\cp_AOA_2.dat"

# File paths for AoA8
cf_file = r"C:\TU_Delft\Master\Thesis\TEMP\AoA8\AOA_8\postProcessing\surfaces\444\cf_AOA_8.dat"
cp_file = r"C:\TU_Delft\Master\Thesis\TEMP\AoA8\AOA_8\postProcessing\surfaces\444\cp_AOA_8.dat"

# File paths for AoA14
cf_file = r"C:\TU_Delft\Master\Thesis\Kasper results models\Thijs results\V3\Re1e6 fully turbulent\AOA_14\postProcessing\surfaces\418\cf_AOA_14.dat"
cp_file = r"C:\TU_Delft\Master\Thesis\Kasper results models\Thijs results\V3\Re1e6 fully turbulent\AOA_14\postProcessing\surfaces\418\cp_AOA_14.dat"

# File paths for AoA-5
cf_file = r"C:\TU_Delft\Master\Thesis\Kasper results models\Thijs results\V3\Re1e6 fully turbulent\AOA_-5\postProcessing\surfaces\2375\cf_AOA_-5.dat"
cp_file = r"C:\TU_Delft\Master\Thesis\Kasper results models\Thijs results\V3\Re1e6 fully turbulent\AOA_-5\postProcessing\surfaces\2375\cp_AOA_-5.dat"

# Load Cf data (skip header)
cf_data = np.loadtxt(cf_file, skiprows=1)
x_cf = cf_data[:, 0]
cfx = cf_data[:, 2]

# Load Cp data (skip header)
cp_data = np.loadtxt(cp_file, skiprows=1)
x_cp = cp_data[:, 0]
cp = cp_data[:, 2]

# Create figure and double axis
fig, ax1 = plt.subplots(figsize=(8, 5))

# Cp plot (left axis)
ax1.plot(x_cp, cp, marker='o', linestyle='-', color='b', label='$C_p$')
# ax1.set_xlabel('x/c')  # Remove x-axis label
ax1.set_ylabel('$C_p$', color='b')
ax1.set_xlim(0, 1)
ax1.set_ylim(-5, 1.5)
ax1.set_yticks(np.arange(-5, 1.6, 1))
ax1.invert_yaxis()
ax1.tick_params(axis='y', labelcolor='b')
ax1.tick_params(axis='x', bottom=False, labelbottom=False)  # Remove x-axis ticks and labels

# Cf plot (right axis)
ax2 = ax1.twinx()
ax2.plot(x_cf, cfx, marker='o', linestyle='-', color='r', label='$C_{fx}$')
ax2.set_ylabel('$C_{fx}$', color='r')
ax2.set_ylim(-0.010, 0.025)
ax2.set_yticks(np.arange(-0.01, 0.026, 0.005))
ax2.tick_params(axis='y', labelcolor='r')
ax2.tick_params(axis='x', bottom=False, labelbottom=False)  # Remove x-axis ticks and labels

# Title and layout
# plt.title('Pressure and Skin Friction Coefficient Distribution')
fig.tight_layout()

# Save as vector file
plt.savefig(r'C:\TU_Delft\Master\Thesis\TEMP\Cp_Cfx_distribution_AoA-5.svg', format='svg')
plt.show()
