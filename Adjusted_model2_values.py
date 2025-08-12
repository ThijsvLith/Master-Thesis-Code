
import pandas as pd
import os
model = 'Model2'
casenames = ['Model2_no_zz_Re_1e6']


file = os.path.join(model, casenames[0], 'unc_'+casenames[0]+'.txt')
print(file)
# Load the file (adjust the path and delimiter if needed)
df = pd.read_csv(file,  sep='\t', header=None, usecols = range(145), dtype=float)

# Filter rows where the 9th column (index 8) is between 340 and 380
df_filtered = df[(df[9] >= 340) & (df[9] <= 380)]

# "C:\TU_Delft\Master\Thesis\Wind tunnel analysis code\Thijs LTT\Test_TvL\Model2\Model2_no_zz_Re_1e6\unc_Model2_no_zz_Re_1e6.txt"
