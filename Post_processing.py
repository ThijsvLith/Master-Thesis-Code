# from Conditions import get_parameters

# case = 'Model2_no_zz_Re_5e5'
# parameters = get_parameters(case)
# print(f"Parameters for {case}: {parameters}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Union
import os
from icecream import ic
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline

def load_and_preprocess_data_unc(filename, model, CPwu_column, CPwl_column, PARAM, z_min=340, z_max=380, alpha_col=1, tol=0.05):
    # Skip the first two rows
    data = pd.read_csv(filename, sep='\t', header=None, skiprows=2, usecols=range(145))
    data = data.apply(pd.to_numeric, errors='coerce')
    PARAM['M'] = data[12].mean()
    if model == 'Model2':
        data = data[((data[9] >= z_min) & (data[9] <= z_max)) | (data[9] == 260)]
        data['alpha_bin'] = data[alpha_col].round(1)
        grouped = data.groupby('alpha_bin', as_index=False, dropna=False).mean()
        grouped = grouped.drop(columns=['alpha_bin'])
        data = grouped
        idx = data[data[9] == 260].index
        ic(idx)
        # --- Added logic here ---
        if not idx.empty:
            use_idx = idx[0]
            if use_idx == 0 and len(idx) > 1:
                use_idx = idx[1]
            PARAM['amax'] = data.iloc[use_idx, alpha_col]
        else:
            PARAM['amax'] = 50
    elif model == 'V3':
        ic(data)
        idx = data[data[9].isin([40,120])].index
        ic(idx)
        # --- Added logic here ---
        if not idx.empty:
            use_idx = idx[0]
            if use_idx == 0 and len(idx) > 1:
                use_idx = idx[1]
            PARAM['amax'] = data.iloc[use_idx, alpha_col]
        else:
            PARAM['amax'] = 50
    keep_cols = [1, 5, 6, 7] + CPwu_column + CPwl_column
    keep_cols = sorted(set(keep_cols))
    data = data.iloc[:, keep_cols]
    return data

def load_and_preprocess_data_corr(filename, model, CPwu_column, CPwl_column, z_min=340, z_max=380, alpha_col=1, tol=0.05):
    # Skip the first two rows
    data = pd.read_csv(filename, sep='\t', header=None, skiprows=2, usecols=range(18))
    # Convert all data to numeric, set errors to NaN
    data = data.apply(pd.to_numeric, errors='coerce')
    if model == 'Model2':
        data = data[(data[13] >= z_min) & (data[13] <= z_max) | (data[13] == 260)]
        data['alpha_bin'] = data[alpha_col].round(1)
        grouped = data.groupby('alpha_bin', as_index=False, dropna=False).mean()
        grouped = grouped.drop(columns=['alpha_bin'])
        data = grouped
    # For V3, just use the data as is (already numeric)
    # Columns to keep: 1, 4, 5, 6, and all in CPwu_column and CPwl_column
    ## ORDER IS alpha, cm, cd, cl, cpwu and cpwl
    keep_cols = [1, 2, 3, 4]
    keep_cols = sorted(set(keep_cols))
    data = data.iloc[:, keep_cols]
    return data

def parse_case_info(case_string):
    parts = case_string.split('_')
    model = parts[0]  # Always the first part
    Re = None

    # Find the part that starts with 'Re'
    for part in parts:
        if part.startswith('Re'):
            Re = part.split('Re')[1]  # e.g., from 'Re1e6' or just '1e6'
            if not Re:  # Handle format like 'Re_1e6'
                Re_index = parts.index(part)
                Re = parts[Re_index + 1]
            break

    return model, Re

def get_cm_from_alpha(alpha_query, Model, Re):
    if Model == 'V3' and Re == '1e6':
        alpha_data = np.array([-10, -5, -2, 0, 2, 6, 10, 12, 14, 16, 18, 20, 22])
        cm_data = np.array([5.90E-02, 5.43E-02, 2.99E-02, -1.34E-02, -5.34E-02, -8.27E-02,
                            -7.64E-02, -8.01E-02, -1.19E-01, -1.48E-01, -1.61E-01,
                            -1.69E-01, -1.75E-01])

    elif Model == 'Model2' and Re == '1e6':
        alpha_data = np.array([-10, -5, -2, 0, 2, 6, 10, 12, 14, 16, 18, 20, 22])
        cm_data = np.array([4.89E-02, 3.20E-02, 1.30E-02, -4.21E-02, -8.64E-02, -1.10E-01,
                            -1.07E-01, -1.06E-01, -1.17E-01, -1.21E-01, -1.35E-01,
                            -1.49E-01, -1.63E-01])
        
    # Spline interpolation with linear extrapolation
    spline = InterpolatedUnivariateSpline(alpha_data, cm_data, k=3, ext='extrapolate')

    # Clip angle of attack to safe extrapolation range
    alpha_clipped = np.clip(alpha_query, -10, 25)
    # Return interpolated/extrapolated Cm
    return spline(alpha_clipped)

# Airfoil input
model = 'Model2'

casename = 'V3_bottom_45deg_0.03c_top_Re_1e6'
casename1 = casename  # You can change this if needed

Method = 'Fmincon'
alpha_min = 2
alpha_max = 8

# Enter columns in which upper and lower pressures are defined
STRIPS = {}
STRIPS['CPwu_column'] = list(range(101, 123))  # Python range is exclusive at the end
STRIPS['CPwl_column'] = list(range(123, 145))
STRIPS['x'] = np.array([0, 178, 333, 468, 585, 685, 775, 851, 918, 975, 1025, 1070, 1120, 1177, 1244, 1320, 1410, 1510, 1627, 1762, 1917, 2095]) / 1000

# Colormap (as an array of RGB values scaled to [0,1])
CC = np.array([
    [66, 146, 198], [22, 61, 90], [239, 59, 44], [140, 140, 140], [255, 204, 86],
    [83, 166, 157], [22, 61, 90], [191, 70, 68], [140, 140, 140], [255, 204, 86],
    [66, 146, 198], [22, 61, 90], [239, 59, 44], [140, 140, 140], [255, 204, 86],
    [83, 166, 157], [22, 61, 90], [191, 70, 68], [140, 140, 140], [255, 204, 86]
]) / 255.0
icolor = 1

# --- Find post-process conditions ---
# from Conditions import get_parameters  # You must define this function in Conditions_Thijs.py
from Determine_Correction_Factor_with_definitions import correction_factor_single_case

model, Re = parse_case_info(casename)

if model == 'Model2':
        PARAM = {
            't_c': 0.0078,      # Thickness over chord
            'lambda': 0.0341,   # NOTE unsure if this is correct, check with the model
            'c': 0.5,           # Chord length
            'c1': 0.5,           # Chord length
            'h': 1.656,          # Height of the wind tu0nnel
            'amin' : -50       # Minimum angle of attack
        }

elif model == 'V3':
    PARAM = {
        't_c': 0.1066,      # Thickness over chord
        'lambda': 0.0539,   # NOTE unsure if this is correct, check with the model
        'c': 0.51948,       # Chord length
        'c1': 0.51948,      # Chord length
        'h': 1.656,          # Height of the wind tunnel
        'amin': -50        # Minimum angle of attack   
    }

## # --- Extract data from files and preprocess ---
if model == 'Model2':
    filename = os.path.join(model, casename, 'unc_all_'+casename+'.txt')
    data_unc = load_and_preprocess_data_unc(filename, model, STRIPS['CPwu_column'], STRIPS['CPwl_column'], PARAM)
elif model == 'V3':
    filename = os.path.join(model, casename, 'unc_'+casename+'.txt')
    data_unc = load_and_preprocess_data_unc(filename, model, STRIPS['CPwu_column'], STRIPS['CPwl_column'], PARAM)


if model == 'Model2':
    filename = os.path.join(model, casename, 'corr_all_'+casename+'.txt')
    data_corr = load_and_preprocess_data_corr(filename, model, STRIPS['CPwu_column'], STRIPS['CPwl_column'])
elif model == 'V3':
    filename = os.path.join(model, casename, 'corr_'+casename+'.txt')
    data_corr = load_and_preprocess_data_corr(filename, model, STRIPS['CPwu_column'], STRIPS['CPwl_column'])

## # --- Calculate correction factor wall pressure strips
PARAM['factor'] = correction_factor_single_case(casename, model, STRIPS, alpha_min, alpha_max, Method, provide_plot=False)

# --- Calculated parameters from input ---
# Standard
PARAM['c_h'] = PARAM['c1'] / PARAM['h']
PARAM['tau'] = 0.25 * PARAM['c_h']
PARAM['sigma'] = np.pi**2 / 48 * PARAM['c_h']**2
PARAM['eps'] = PARAM['sigma'] * PARAM['lambda']
PARAM['beta'] = np.sqrt(1 - PARAM['M']**2)
PARAM['eps_alpha'] = 1.1 * PARAM['beta'] / PARAM['t_c']

# For generally attached flow
PARAM['factor_sc'] = PARAM['sigma'] / PARAM['beta']**2                  # streamline curvature
PARAM['factor_sb'] = PARAM['lambda'] * PARAM['sigma'] / PARAM['beta']**3 # solid blockage
PARAM['factor_wb'] = PARAM['tau'] * (1 + 0.4 * PARAM['M']**2) / PARAM['beta']**2 # wake blockage

# Make the Cm column zero in both dataframes
data_unc.iloc[:, 3] = 0
data_corr.iloc[:, 3] = 0

## # --- Remove zero-lift from pressure data and lift coefficient ---

# Copy data_unc to POSTDATA_A_presscorr
POSTDATA_A_presscorr = data_unc.copy().values  # as numpy array for easier indexing

# Extract zero-lift pressure (interpolate all columns 5:end at Cl=0)

# MATLAB columns 3 and 5:end -> Python columns 2 and 4: (0-based)
cl_col = 2
press_cols = np.arange(4, POSTDATA_A_presscorr.shape[1])

# Interpolator for each pressure column
zerolift_press = np.array([
    interp1d(POSTDATA_A_presscorr[:, cl_col], POSTDATA_A_presscorr[:, col], kind='linear', fill_value='extrapolate')(0)
    for col in press_cols
])

# Subtract zero-lift pressure from all rows, columns 5:end
POSTDATA_A_presscorr[:, 4:] = POSTDATA_A_presscorr[:, 4:] - zerolift_press


# Update lift with the new pressure data
n_x = len(STRIPS['x'])
Cpwu = POSTDATA_A_presscorr[:, 4:4+n_x]
Cpwl = POSTDATA_A_presscorr[:, 4+n_x:4+2*n_x]
diffCp = Cpwl - Cpwu

# Trapz integration along STRIPS['x'] for each row. This calculates the new lift coefficient based on the pressures corrected for zero-lift pressure
POSTDATA_A_presscorr[:, 2] = np.trapz(diffCp, STRIPS['x'], axis=1) * (1 / PARAM['c'])

## # --- Apply wallstrip correction factor determined by the script Determine_Correction_Factor.py ---
POSTDATA_B_stripcorr = POSTDATA_A_presscorr.copy()
POSTDATA_B_stripcorr[:, 2] = POSTDATA_B_stripcorr[:, 2] * PARAM['factor']

# Copy B_stripcorr to C1_cdcorr
POSTDATA_C1_cdcorr = POSTDATA_B_stripcorr.copy()

## # --- Extrapolate the Cd values when the wake rake cuts out
# Create logical mask for "bad" values

logical = (
    (POSTDATA_C1_cdcorr[:, 0] <= PARAM['amin']) |
    (POSTDATA_C1_cdcorr[:, 0] >= PARAM['amax']) |
    (POSTDATA_C1_cdcorr[:, 1] <= 0.005)
)

# Inverse mask for "good" values
antilogical = ~logical

interp_func = interp1d(
    POSTDATA_C1_cdcorr[antilogical, 0],  # x: alpha (good)
    POSTDATA_C1_cdcorr[antilogical, 1],  # y: Cl or Cd (good)
    kind='linear', ## can be QUADRATIC
    fill_value='extrapolate'
)

POSTDATA_C1_cdcorr[logical, 1] = interp_func(POSTDATA_C1_cdcorr[logical, 0])


## # --- Addd guestimate for moment
## Interpolate the data from Kasper's model. Then get the Cm per AoA in the dataframe in the fourth column.
## This can be used to correct the angle of attack in the future.
# This will be done later

## Do make the first copy of the dataframe so code is future-proof
POSTDATA_C2_cmcorr = POSTDATA_C1_cdcorr.copy()

POSTDATA_C2_cmcorr[:,3] = get_cm_from_alpha(POSTDATA_C2_cmcorr[:,0], model, Re)


## # --- Re apply all wind tunnel correction factors.

## NOTE: POSTDATA_D1_wallcorr is the data WITHOUT extrpolation of the Cd and Cm values.
## NOTE: POSTDATA_D2_wallcorr is the data WITH extrapolation of the Cd and Cm values.

# Apply wall correction and Cd correction
POSTDATA_D1_wallcorr = POSTDATA_B_stripcorr.copy()
POSTDATA_D2_wallcdcmcorr = POSTDATA_C2_cmcorr.copy()

# Wake blockage
PARAM['D1_wake_blockage'] = PARAM['factor_wb'] * POSTDATA_D1_wallcorr[:, 1]
PARAM['D2_wake_blockage'] = PARAM['factor_wb'] * POSTDATA_D2_wallcdcmcorr[:, 1]

# Solid blockage
PARAM['eps_sol_total'] = PARAM['factor_sb'] * (1 + PARAM['eps_alpha'] * (POSTDATA_D1_wallcorr[:, 0] * np.pi / 180) ** 2)
PARAM['eps_total_plusmach'] = (2 - PARAM['M'] ** 2) * (PARAM['D1_wake_blockage'] + PARAM['eps_sol_total'])

# Alpha correction
POSTDATA_D1_wallcorr[:, 0] = POSTDATA_D1_wallcorr[:, 0] + 57.3 * PARAM['sigma'] * (POSTDATA_D1_wallcorr[:, 2] + 4 * POSTDATA_D1_wallcorr[:, 3]) / (PARAM['beta'] * 2 * np.pi)
POSTDATA_D2_wallcdcmcorr[:, 0] = POSTDATA_D2_wallcdcmcorr[:, 0] + 57.3 * PARAM['sigma'] * (POSTDATA_D2_wallcdcmcorr[:, 2] + 4 * POSTDATA_D2_wallcdcmcorr[:, 3]) / (PARAM['beta'] * 2 * np.pi)

# Cm correction NOTE THE FIRST LINE SHOULD NOT BE MINUS
PARAM['streamline_curvature_cm'] = -POSTDATA_D1_wallcorr[:, 2] * (PARAM['sigma'] / PARAM['beta'] ** 2 / 4 - 1.05 * (PARAM['sigma'] / PARAM['beta'] ** 2) ** 2)
POSTDATA_D1_wallcorr[:, 3] = POSTDATA_D1_wallcorr[:, 3] * (1 - PARAM['eps_total_plusmach']) + PARAM['streamline_curvature_cm']
POSTDATA_D2_wallcdcmcorr[:, 3] = POSTDATA_D2_wallcdcmcorr[:, 3] * (1 - PARAM['eps_total_plusmach']) + PARAM['streamline_curvature_cm']

# Cl correction
PARAM['streamline_curvature_cl'] = -POSTDATA_D1_wallcorr[:, 2] * (PARAM['sigma'] / PARAM['beta'] ** 2 - 5.25 * (PARAM['sigma'] / PARAM['beta'] ** 2) ** 2)
PARAM['solid_blockage_cl'] = -POSTDATA_D1_wallcorr[:, 2] * (2 - PARAM['M'] ** 2) * PARAM['eps_sol_total']
PARAM['D1_wake_blockage_cl'] = -POSTDATA_D1_wallcorr[:, 2] * (2 - PARAM['M'] ** 2) * PARAM['D1_wake_blockage']
PARAM['D2_wake_blockage_cl'] = -POSTDATA_D2_wallcdcmcorr[:, 2] * (2 - PARAM['M'] ** 2) * PARAM['D2_wake_blockage']
POSTDATA_D1_wallcorr[:, 2] = POSTDATA_D1_wallcorr[:, 2] + PARAM['streamline_curvature_cl'] + PARAM['D1_wake_blockage_cl'] + PARAM['solid_blockage_cl']
POSTDATA_D2_wallcdcmcorr[:, 2] = POSTDATA_D2_wallcdcmcorr[:, 2] + PARAM['streamline_curvature_cl'] + PARAM['D2_wake_blockage_cl'] + PARAM['solid_blockage_cl']

# Cd correction
PARAM['D1_wake_blockage_cd'] = -POSTDATA_D1_wallcorr[:, 1] * (2 - PARAM['M'] ** 2) * PARAM['D1_wake_blockage']
PARAM['D1_solid_blockage_cd'] = -POSTDATA_D1_wallcorr[:, 1] * (2 - PARAM['M'] ** 2) * PARAM['eps_sol_total']
POSTDATA_D1_wallcorr[:, 1] = POSTDATA_D1_wallcorr[:, 1] + PARAM['D1_wake_blockage_cd'] + PARAM['D1_solid_blockage_cd']
PARAM['D2_wake_blockage_cd'] = -POSTDATA_D2_wallcdcmcorr[:, 1] * (2 - PARAM['M'] ** 2) * PARAM['D2_wake_blockage']
PARAM['D2_solid_blockage_cd'] = -POSTDATA_D2_wallcdcmcorr[:, 1] * (2 - PARAM['M'] ** 2) * PARAM['eps_sol_total']
POSTDATA_D2_wallcdcmcorr[:, 1] = POSTDATA_D2_wallcdcmcorr[:, 1] + PARAM['D2_wake_blockage_cd'] + PARAM['D2_solid_blockage_cd']


## # --- VALUES FROM KASKPERS MODEL ---

# alpha_kasper = [-10, -5, -2, 0, 2, 6, 10, 12, 14, 16, 18, 20, 22]
# # Cl_kasper = [-0.27, -0.152, -0.0115, 0.254, 0.59, 1.14, 1.53, 1.68, 1.73, 1.16, 1.08, 1.07, 1.07]
# Cl_kasper = [-0.28, -0.113, 0.0202, 0.323, 0.686, 1.24, 1.65, 1.81, 1.91, 1.52, 1.45, 1.39, 1.31]
# Cd_kasper = [0.115, 0.071, 0.0505, 0.0366, 0.0275, 0.0161, 0.025, 0.0383, 0.0828, 0.185, 0.241, 0.287, 0.323]


## # --- Plotting the results ---
fig = plt.figure(figsize=(12, 4))  # Similar to [100 100 1200 360] in pixels

# First subplot: Lift polar
ax1 = fig.add_axes([0.08, 0.2, 0.4, 0.72])
ax1.grid(True)
ax1.tick_params(labelsize=18)
ax1.plot(data_unc.values[:, 0], data_unc.values[:, 2], '-x', linewidth=2, color=CC[0], label='unc')
ax1.plot(POSTDATA_D1_wallcorr[:, 0], POSTDATA_D1_wallcorr[:, 2], '-x', linewidth=2, color=CC[1], label='corr')
ax1.plot(POSTDATA_D2_wallcdcmcorr[:, 0], POSTDATA_D2_wallcdcmcorr[:, 2], '-x', linewidth=2, color=CC[2], label='corr extrap')
ax1.plot(data_corr.values[:, 0], data_corr.values[:, 2], '--', linewidth=2, color='k', label='old corr - ref')
# ax1.plot(alpha_kasper, Cl_kasper, '--', linewidth=2, color='g', label='Kasper model')
ax1.set_xlabel(r'Angle of attack, $\alpha$ [deg]', fontsize=18)
ax1.set_ylabel(r'Lift coefficient, $C_l$ [-]', fontsize=18)
ax1.set_title(f'Lift polar - {casename1}', fontsize=18)

# Second subplot: Drag polar
ax2 = fig.add_axes([0.58, 0.2, 0.4, 0.72])
ax2.grid(True)
ax2.tick_params(labelsize=18)
ax2.plot(data_unc.values[:, 0], data_unc.values[:, 1], '-x', linewidth=2, color=CC[0], label='unc')
ax2.plot(POSTDATA_D1_wallcorr[:, 0], POSTDATA_D1_wallcorr[:, 1], '-x', linewidth=2, color=CC[1], label='corr')
ax2.plot(POSTDATA_D2_wallcdcmcorr[:, 0], POSTDATA_D2_wallcdcmcorr[:, 1], '-x', linewidth=2, color=CC[2], label='corr extrap')
ax2.plot(data_corr.values[:, 0], data_corr.values[:, 1], '--', linewidth=2, color='k', label='old corr - ref')
# ax2.plot(alpha_kasper, Cd_kasper, '--', linewidth=2, color='g', label='Kasper model')
ax2.set_xlabel(r'Angle of attack, $\alpha$ [deg]', fontsize=18)
ax2.set_ylabel(r'Drag coefficient, $C_d$ [-]', fontsize=18)
ax2.set_title(f'Drag polar - {casename1}', fontsize=18)
ax2.legend(loc='upper left', fontsize=14)

plt.show()

# ic(POSTDATA_D2_wallcdcmcorr[:, 0])  # Print the angle of attack values
# ic(POSTDATA_D2_wallcdcmcorr[:, 2])  # Print the lift coefficient values

