import numpy as np
import matplotlib.pyplot as plt

# File paths
cf_file = r"C:\TU_Delft\Master\Thesis\Kasper results models\V3_results_Re1\OF_Results\OF_Cf\cf_Re1000000_t0.0782_cx0.178_cy0.0953_r0_LE0.351_camTE0.1_a2.dat"
cp_file = r"C:\TU_Delft\Master\Thesis\Kasper results models\V3_results_Re1\OF_Results\OF_Cp\cp_Re1000000_t0.0782_cx0.178_cy0.0953_r0_LE0.351_camTE0.1_a2.dat"

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
ax1.set_xlabel('x/c')
ax1.set_ylabel('$C_p$', color='b')
ax1.set_xlim(0, 1)
ax1.invert_yaxis()  # Invert y-axis for Cp plot
ax1.tick_params(axis='y', labelcolor='b')
# ax1.grid(True)

# Cf plot (right axis)
ax2 = ax1.twinx()
ax2.plot(x_cf, cfx, marker='o', linestyle='-', color='r', label='$C_{fx}$')
ax2.set_ylabel('$C_{fx}$', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Title and layout
# plt.title('Pressure and Skin Friction Coefficient Distribution')
fig.tight_layout()

# Save as vector file
plt.savefig(r'C:\TU_Delft\Master\Thesis\TEMP\Cp_Cfx_distribution.svg', format='svg')
plt.show()
