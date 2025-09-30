##Hysteresis analysis
import matplotlib.pyplot as plt

import pandas as pd

from icecream import ic

filename = r"C:\TU_Delft\Master\Thesis\Wind tunnel analysis code\Thijs LTT\Test_TvL\Model2\Test runs\unc_test_run1-hysteresis.txt"

data = pd.read_csv(filename, sep='\t', header=None, skiprows=2, usecols=range(145))

ic(data)
plt.figure(figsize=(10, 6))
plt.plot(data.iloc[:21,1], data.iloc[:21,6], 'o-', label = 'Positive delta AoA')
plt.plot(data.iloc[20:,1], data.iloc[20:,6], 'o-', label = 'Negative delta AoA')

plt.xlabel(r'$\alpha$ ($^\circ$)')
plt.ylabel(r'$C_l$ (-)')
plt.xlim(0, 20)
plt.xticks(range(0, 21, 5))
plt.ylim(0, 2)
plt.yticks([0, 0.5, 1.0, 1.5, 2.0])
# plt.title('Hysteresis effect on Cl')
plt.tight_layout()
plt.grid()
plt.legend()
plt.show()