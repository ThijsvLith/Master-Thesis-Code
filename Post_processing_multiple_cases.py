# from Conditions import get_parameters

"""
Post-processing script for wind tunnel wind tunnel analysis.

This script loads, processes, and visualizes aerodynamic data for multiple wind tunnel cases.
It includes:
    - Data loading and preprocessing for Model2 and V3 cases
    - Correction factor calculation and application
    - Extrapolation and uncertainty estimation
    - Plotting of lift, drag, and other aerodynamic coefficients

Functions:
    - concise_label: Generate concise legend labels from case names
    - load_and_preprocess_data_unc: Load and preprocess uncorrected data
    - load_and_preprocess_data_corr: Load and preprocess corrected data
    - parse_case_info: Parse model and Reynolds number from case name
    - get_cm_from_alpha: Interpolate/extrapolate moment coefficient from Kasper's model

Usage:
    Adjust the 'casenames' list and STRIPS dictionary as needed for your cases.
    Run the script to generate summary plots for all cases.
"""

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

def concise_label(casename):
    """
    Generate a concise label for the given case name by removing model and Reynolds number info.

    Args:
        casename (str): Full case name string.
    Returns:
        str: Concise label for legend.
    """
    label = casename.replace('Model2_', '').replace('V3_', '')
    label = label.replace('_Re_5e5', '').replace('_Re_1e6', '')
    return label

def load_and_preprocess_data_unc(filename, model, CPwu_column, CPwl_column, PARAM, z_min=340, z_max=380, alpha_col=1, tol=0.05):
    """
    Load and preprocess uncorrected wind tunnel data for a given case.
    Applies filtering, grouping, and extracts relevant columns.

    Args:
        filename (str): Path to data file.
        model (str): Model name ('Model2' or 'V3').
        CPwu_column (list): Indices for upper pressure taps.
        CPwl_column (list): Indices for lower pressure taps.
        PARAM (dict): Dictionary to store parameters and results.
        z_min (float): Minimum z value for filtering.
        z_max (float): Maximum z value for filtering.
        alpha_col (int): Column index for angle of attack.
        tol (float): Tolerance for grouping.
    Returns:
        pd.DataFrame: Preprocessed data.
    """
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
        PARAM['skip_idx'] = idx.tolist()
        ic(idx)
        # --- Added logic here ---
        if not idx.empty:
            use_idx = idx[0]
            if use_idx == 0:
                if len(idx) > 1:
                    use_idx = idx[1]
                    PARAM['amax'] = data.iloc[use_idx, alpha_col]
                    PARAM['amax_idx'] = use_idx
                else:
                    PARAM['amax'] = 50
            else:
                PARAM['amax'] = data.iloc[use_idx, alpha_col]
                PARAM['amax_idx'] = use_idx
        else:
            PARAM['amax'] = 50

    elif model == 'V3':
        idx = data[data[9].isin([40,120])].index
        PARAM['skip_idx'] = idx.tolist()
        # --- Added logic here ---
        if not idx.empty:
            use_idx = idx[0]
            if use_idx == 0:
                if len(idx) > 1:
                    use_idx = idx[1]
                    PARAM['amax'] = data.iloc[use_idx, alpha_col]
                    PARAM['amax_idx'] = use_idx
                else:
                    PARAM['amax'] = 50
            else:
                PARAM['amax'] = data.iloc[use_idx, alpha_col]
                PARAM['amax_idx'] = use_idx
        else:
            PARAM['amax'] = 50
    keep_cols = [1, 5, 6, 7] + CPwu_column + CPwl_column
    keep_cols = sorted(set(keep_cols))
    data = data.iloc[:, keep_cols]
    
    return data

def load_and_preprocess_data_corr(filename, model, CPwu_column, CPwl_column, z_min=340, z_max=380, alpha_col=1, tol=0.05):
    """
    Load and preprocess corrected wind tunnel data for a given case.
    Applies filtering, grouping, and extracts relevant columns.

    Args:
        filename (str): Path to data file.
        model (str): Model name ('Model2' or 'V3').
        CPwu_column (list): Indices for upper pressure taps.
        CPwl_column (list): Indices for lower pressure taps.
        z_min (float): Minimum z value for filtering.
        z_max (float): Maximum z value for filtering.
        alpha_col (int): Column index for angle of attack.
        tol (float): Tolerance for grouping.
    Returns:
        pd.DataFrame: Preprocessed data.
    """
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
    """
    Parse the model name and Reynolds number from a case name string.

    Args:
        case_string (str): Full case name string.
    Returns:
        tuple: (model, Re) where model is str and Re is str or None.
    """
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
    """
    Interpolate or extrapolate the moment coefficient (Cm) for a given angle of attack
    using Kasper's model data and spline interpolation.

    Args:
        alpha_query (array-like): Angles of attack to query.
        Model (str): Model name ('Model2' or 'V3').
        Re (str): Reynolds number as string.
    Returns:
        np.ndarray: Interpolated/extrapolated Cm values.
    """
    ## THESE VALUES ARE CMPitch From Kaspers model fully turbulent
    if Model == 'V3' and Re == '1e6':
        alpha_data = np.array([-10, -5, -2, 0, 2, 6, 10, 12, 14, 16, 18, 20, 22])
        cm_data = np.array([5.90E-02, 5.43E-02, 2.99E-02, -1.34E-02, -5.34E-02, -8.27E-02,
                            -7.64E-02, -8.01E-02, -1.19E-01, -1.48E-01, -1.61E-01,
                            -1.69E-01, -1.75E-01])

    elif Model == 'V3' and Re == '5e5':
        alpha_data = np.array([-2, 6, 10, 12, 14, 16, 18, 20, 22])
        cm_data = np.array([1.86E-02,-8.64E-02,-7.86E-02,-8.82E-02,-1.89E-01,-1.49E-01, -1.55E-01, -1.52E-01, -1.57E-01
                            ])
        
    elif Model == 'Model2' and Re == '1e6':
        alpha_data = np.array([-10, -5, -2, 0, 2, 6, 10, 12, 14, 16, 18, 20, 22])
        cm_data = np.array([4.89E-02, 3.20E-02, 1.30E-02, -4.21E-02, -8.64E-02, -1.10E-01,
                            -1.07E-01, -1.06E-01, -1.17E-01, -1.21E-01, -1.35E-01,
                            -1.49E-01, -1.63E-01])
        
    elif Model == 'Model2' and Re == '5e5':
        alpha_data = np.array([6, 10, 12, 14, 16, 18, 20, 22])
        cm_data = np.array([-1.03E-01, -9.24E-02, -9.58E-02, -1.23E-01, -1.69E-01, -1.74E-01, -1.81E-01, -1.86E-01
                            ])
        
    # Spline interpolation with linear extrapolation
    spline = InterpolatedUnivariateSpline(alpha_data, cm_data, k=3, ext='extrapolate')

    # Clip angle of attack to safe extrapolation range
    alpha_clipped = np.clip(alpha_query, -10, 25)
    # Return interpolated/extrapolated Cm
    return spline(alpha_clipped)

# Airfoil input
# model = 'Model2'

casenames = ['Model2_no_zz_Re_5e5',
             'Model2_small_zz_bottom_Re_5e5',
             'Model2_zz_0.1c_top_Re_5e5',
             'Model2_zz_0.05c_top_Re_5e5',
             'Model2_zz_bottom_0.05c_top_Re_5e5',
             'Model2_zz_bottom_Re_5e5']  # State all different cases here

# casenames = ['Model2_no_zz_Re_1e6', 
#              'Model2_small_zz_bottom_Re_1e6',
#             'Model2_zz_0.1c_top_Re_1e6', 
#             'Model2_zz_0.05c_top_Re_1e6',
#             'Model2_zz_bottom_0.05c_top_Re_1e6',
#             'Model2_zz_bottom_Re_1e6']  ## State all different cases here

# casenames = ['V3_no_zz_Re_5e5',
#             'V3_small_zz_bottom_Re_5e5',
#             'V3_zz_0.05c_top_Re_5e5',
#             'V3_zz_bottom_0.05c_top_Re_5e5', 
#             'V3_bottom_45_deg_Re_5e5']

# casenames = ['V3_no_zz_Re_1e6', 
#                     'V3_small_zz_bottom_Re_1e6', 
#                     'V3_zz_0.05c_top_Re_1e6', 
#                     'V3_zz_bottom_0.05c_top_Re_1e6', 
#                     'V3_bottom_0.03c_top_Re_1e6',
#                     'V3_bottom_45deg_0.03c_top_Re_1e6', 
#                     'V3_bottom_45_deg_Re_1e6']

# casenames = ['V3_no_zz_Re_5e5']

Method = 'Fmincon'
alpha_min = 2
alpha_max = 10

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

# Create one figure with three subplots side by side
fig, axs = plt.subplots(2, 2, figsize=(18, 10))
ax1, ax2 = axs[0, 0], axs[0, 1]
ax3, ax4 = axs[1, 0], axs[1, 1]
ax1.grid(True)
ax1.tick_params(labelsize=14)
ax2.grid(True)
ax2.tick_params(labelsize=14)
ax3.grid(True)
ax3.tick_params(labelsize=14)
ax4.grid(True)
ax4.tick_params(labelsize=14)

for casename in casenames:

    model, Re = parse_case_info(casename)
    print(model, Re)

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
        filename_corr = os.path.join(model, casename, 'corr_all_'+casename+'.txt')
        data_corr = load_and_preprocess_data_corr(filename_corr, model, STRIPS['CPwu_column'], STRIPS['CPwl_column'])
    elif model == 'V3':
        filename = os.path.join(model, casename, 'unc_'+casename+'.txt')
        data_unc = load_and_preprocess_data_unc(filename, model, STRIPS['CPwu_column'], STRIPS['CPwl_column'], PARAM)
        filename_corr = os.path.join(model, casename, 'corr_'+casename+'.txt')
        data_corr = load_and_preprocess_data_corr(filename_corr, model, STRIPS['CPwu_column'], STRIPS['CPwl_column'])

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
    ic(PARAM)
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

    # Only plot POSTDATA_D1_wallcorr for each case
    # color = CC[i % len(CC)]

    if 'skip_idx' in PARAM and PARAM['skip_idx']:
        skip_idx = sorted(PARAM['skip_idx'])
        full_data = POSTDATA_D2_wallcdcmcorr
        mask = np.ones(full_data.shape[0], dtype=bool)
        mask[skip_idx] = False

        # Split corrected and skipped data
        data_corrected = full_data[mask, :]
        data_skipped = full_data[skip_idx, :]

        # Plot corrected Cl (solid)
        label = concise_label(casename)
        line = ax1.plot(data_corrected[:, 0], data_corrected[:, 2], '-', linewidth=2, label=label)[0]
        dash_color = line.get_color()

        # Plot dashed segments for skipped Cl values
        for idx in skip_idx:
            if idx == 0:
                segment = full_data[[idx], :]
            else:
                prev_idx = idx - 1
                segment = full_data[[prev_idx, idx], :]
            ax1.plot(segment[:, 0], segment[:, 2], ':', linewidth=2, color=dash_color, label='_nolegend_')

        # amax marker
        amax_idx = PARAM.get('amax_idx', None)
        if amax_idx is not None:
            ax1.plot(
                full_data[amax_idx - 1, 0],
                full_data[amax_idx - 1, 2],
                marker='x',
                markersize=10,
                color=dash_color,
                label='_nolegend_'
            )

        # === Cd vs alpha (no legend) ===
        ax2.plot(data_corrected[:, 0], data_corrected[:, 1], '-', linewidth=2, color=dash_color, label=label)

        # === Cl vs Cd (no legend) ===
        ax3.plot(data_corrected[:, 1], data_corrected[:, 2], '-', linewidth=2, color=dash_color, label=label)

        ax4.plot(data_corrected[:, 0], data_corrected[:, 2]/data_corrected[:, 1], '-', linewidth=2, color=dash_color, label=label)

    else:
        label = concise_label(casename)
        line = ax1.plot(POSTDATA_D2_wallcdcmcorr[:, 0], POSTDATA_D2_wallcdcmcorr[:, 2], '-', linewidth=2, label=label)[0]
        dash_color = line.get_color()
        ax2.plot(POSTDATA_D2_wallcdcmcorr[:, 0], POSTDATA_D2_wallcdcmcorr[:, 1], '-', linewidth=2, color=dash_color, label=label)
        ax3.plot(POSTDATA_D2_wallcdcmcorr[:, 1], POSTDATA_D2_wallcdcmcorr[:, 2], '-', linewidth=2, color=dash_color, label=label)
        ax4.plot(POSTDATA_D2_wallcdcmcorr[:, 0], POSTDATA_D2_wallcdcmcorr[:, 2]/POSTDATA_D2_wallcdcmcorr[:, 1], '-', linewidth=2, color=dash_color, label=label)


# alpha_kasper = [-10, -5, -2, 0, 2, 6, 10, 12, 14, 16, 18, 20, 22]
# Cl_kasper = [-0.27, -0.152, -0.0115, 0.254, 0.59, 1.14, 1.53, 1.68, 1.73, 1.16, 1.08, 1.07, 1.07] ## V3
# # Cl_kasper = [-0.28, -0.113, 0.0202, 0.323, 0.686, 1.24, 1.65, 1.81, 1.91, 1.52, 1.45, 1.39, 1.31] # Model2
# Cd_kasper = [0.115, 0.071, 0.0505, 0.0366, 0.0275, 0.0161, 0.025, 0.0383, 0.0828, 0.185, 0.241, 0.287, 0.323] #V3
# # Cd_kasper = [1.19E-01, 6.60E-02, 4.42E-02, 3.33E-02, 2.38E-02, 1.83E-02, 2.27E-02, 3.18E-02, 5.44E-02, 1.16E-01, 1.60E-01, 2.08E-01, 2.63E-01]



# ########### TESTTEST
# alpha_kasper = [6,10,12,14,16,18,20]
# Cl_kasper = [1.17, 1.5,1.61,1.43,1.09,1.13,1.23]
# Cd_kasper = [0.017, 0.0227, 0.0423, 0.0973, 0.179, 0.194, 0.231]


# ax1.plot(alpha_kasper, Cl_kasper, '--', linewidth=2, color='g', label='Kasper model')
# ax1.set_title(rf'$C_l$–$\alpha$ for {model} at Re = {Re}', fontsize=16)
ax1.set_xlabel(r'$\alpha$ ($^\circ$)', fontsize=18)
ax1.set_ylabel(r'$C_\mathrm{l}$ (-)', fontsize=18)
# ax1.legend(fontsize=12)

# ax2.plot(alpha_kasper, Cd_kasper, '--', linewidth=2, color='g', label='Kasper model')
# ax2.set_title(rf'$C_d$–$\alpha$ for {model} at Re = {Re}', fontsize=16)
ax2.set_xlabel(r'$\alpha$ ($^\circ$)', fontsize=16)
ax2.set_ylabel(r'$C_\mathrm{d}$ (-)', fontsize=16)
# ax2.legend(fontsize=12, loc='upper left')  # Add legend to ax2 in the top left corner

# ax3.plot(Cd_kasper,Cl_kasper, '--', linewidth=2, color='g', label='Kasper model')
# ax3.set_title(rf'$C_l$–$C_d$ for {model} at Re = {Re}', fontsize=16)
ax3.set_xlabel(r'$C_\mathrm{d}$ (-)', fontsize=16)
ax3.set_ylabel(r'$C_\mathrm{l}$ (-)', fontsize=16)

ax4.set_xlabel(r'$\alpha$ ($^\circ$)', fontsize=16)
ax4.set_ylabel(r'$C_\mathrm{l}/C_\mathrm{d}$ (-)', fontsize=16)


## Set same axis size for Cl Cd and Cl/cd - alpha plots
for ax in [ax1, ax2, ax4]:
    ax.set_xlim(-11, 26)
    ax.set_xticks(np.arange(-10, 26, 5))

# Example: Set y-axis limits (adjust as needed for your data)
ax1.set_ylim(-0.5, 2.0)   # Cl axis
ax1.set_yticks(np.arange(-0.5, 2.6, 0.5))

ax2.set_ylim(0, 0.4)      # Cd axis
ax2.set_yticks(np.arange(0, 0.41, 0.1))

ax4.set_ylim(-10, 100)      # Cl/Cd axis
ax4.set_yticks(np.arange(0, 101, 20))

## Set same axis size for Cl-Cd
ax3.set_xlim(0, 0.4)           # Set x-axis limits for Cd
ax3.set_xticks(np.arange(0, 0.41, 0.1))  # Set x-axis ticks for Cd

ax3.set_ylim(-0.5, 2.5)         # Set y-axis limits for Cl
ax3.set_yticks(np.arange(-0.5, 2.6, 0.5)) # Set y-axis ticks for Cl

# Place a single legend at the bottom center of the figure
handles, labels = ax1.get_legend_handles_labels()

plt.subplots_adjust(bottom=0.15)

fig.legend(
    handles,
    labels,
    loc='lower center',
    bbox_to_anchor=(0.45, 0.02),  # Move legend bar up into the figure
    ncol=len(labels),
    fontsize=12
)

fig.savefig(r'C:\TU_Delft\Master\Thesis\Figures overleaf\Results\POLARS\Model2_Re5e5_polars_new.png', dpi=300, bbox_inches='tight')



plt.show()


