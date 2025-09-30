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

def data_cfd(model,Re, transition):
    """
    Load CFD data based on model, Reynolds number, and transition state.

    Args:
        model (str): 'V3' or 'Model2'
        Re (str): '1e6' or '5e5'
        transition (bool): True for transition, False for fully turbulent

    Returns:
        pd.DataFrame: Loaded CFD data
    """
    root_path = r"C:\TU_Delft\Master\Thesis\Kasper results models\Thijs results"
    # Select model folder
    if model.lower() == 'v3':
        model_folder = 'V3'
    elif model.lower() == 'model2':
        model_folder = 'Model2'
    else:
        raise ValueError("Model must be 'V3' or 'Model2'")

    # Select Reynolds number and transition/turbulent folder
    if Re == '1e6':
        if transition:
            data_file = 'Re1e6 transition'
        else:
            data_file = 'Re1e6 fully turbulent'
    elif Re == '5e5':
        if transition:
            data_file = 'Re5e5 transition'
        else:
            data_file = 'Re5e5 fully turbulent'
    else:
        raise ValueError("Re must be '1e6' or '5e5'")

    dat_file = 'polar.dat'

    # Build full file path
    file_path = os.path.join(root_path, model_folder, data_file, dat_file)

    ic(file_path)
    # Load data: first row as header, rest as data
    if os.path.exists(file_path):
        # Read the first row for column names
        with open(file_path, 'r') as f:
            header = f.readline().strip().split()
        # Read the rest of the data
        df = pd.read_csv(file_path, sep='\s+', header=None, skiprows=1, names=header)
        # Remove all rows where iter_n == 5001
        if 'iter_n' in df.columns:
            df = df[df['iter_n'] != 5001]
        
        return df
    else:
        raise FileNotFoundError(f"File not found: {file_path}")

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

def load_and_preprocess_data_unc(filename_unc_all, filename_unc, model, CPwu_column, CPwl_column, PARAM, z_min=340, z_max=380, alpha_col=1, tol=0.05):
    # Skip the first two rows
    data_unc_all = pd.read_csv(filename_unc_all, sep='\t', header=None, skiprows=2, usecols=range(145))
    data_unc = pd.read_csv(filename_unc, sep='\t', header=None, skiprows=2, usecols=range(145))
    data_unc_all = data_unc_all.apply(pd.to_numeric, errors='coerce')
    data_unc = data_unc.apply(pd.to_numeric, errors='coerce')
    PARAM['M'] = data_unc[12].mean()
    if model == 'Model2':
        data_unc_all = data_unc_all[((data_unc_all[9] >= z_min) & (data_unc_all[9] <= z_max)) | (data_unc_all[9] == 260)]
        data_unc_all['alpha_bin'] = data_unc_all[alpha_col].round(1)
        grouped = data_unc_all.groupby('alpha_bin', as_index=False, dropna=False).mean()
        grouped = grouped.drop(columns=['alpha_bin'])
        data = grouped
        idx = data[data[9] == 260].index
        # ic(idx)
        PARAM['skip_idx'] = idx.tolist()
        # --- Added logic here ---
        if not idx.empty:
            use_idx = idx[0]
            if use_idx == 0:
                if len(idx) > 1:
                    use_idx = idx[1]
                    PARAM['amax'] = data.iloc[use_idx, alpha_col]
                else:
                    PARAM['amax'] = 50
            else:
                PARAM['amax'] = data.iloc[use_idx, alpha_col]
        else:
            PARAM['amax'] = 50

    elif model == 'V3':
        idx = data_unc[data_unc[9].isin([40,120])].index
        # ic(idx)
        PARAM['skip_idx'] = idx.tolist()
        # --- Added logic here ---
        if not idx.empty:
            use_idx = idx[0]
            if use_idx == 0:
                if len(idx) > 1:
                    use_idx = idx[1]
                    PARAM['amax'] = data_unc.iloc[use_idx, alpha_col]
                else:
                    PARAM['amax'] = 50
            else:
                PARAM['amax'] = data_unc.iloc[use_idx, alpha_col]
        else:
            PARAM['amax'] = 50
    keep_cols = [1, 5, 6, 7] + CPwu_column + CPwl_column
    keep_cols = sorted(set(keep_cols))
    data = data_unc_all.iloc[:, keep_cols]
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
model = 'Model2'

# casenames = ['Model2_no_zz_Re_5e5',
#              'Model2_small_zz_bottom_Re_5e5',
#              'Model2_zz_0.1c_top_Re_5e5',
#              'Model2_zz_0.05c_top_Re_5e5',
#              'Model2_zz_bottom_0.05c_top_Re_5e5',
#              'Model2_zz_bottom_Re_5e5']  # State all different cases here

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

# casenames = ['V3_bottom_45deg_0.03c_top_Re_1e6']

# casenames = ['Model2_no_zz_Re_5e5']

# casenames = ['Model2_no_zz_Re_1e6']

casenames = ['V3_no_zz_Re_1e6']

# casenames = ['V3_no_zz_Re_5e5']


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

# # Prepare the plot
fig = plt.figure(figsize=(22, 7))  # Similar to [100 100 1200 360] in pixels

# First subplot: Lift polar
ax1 = fig.add_axes([0.08, 0.2, 0.4, 0.72])
ax1.grid(True)
ax1.tick_params(labelsize=18)

# Second subplot: Drag polar
ax2 = fig.add_axes([0.58, 0.2, 0.4, 0.72])
ax2.grid(True)
ax2.tick_params(labelsize=18)



# --- Find post-process conditions ---
# from Conditions import get_parameters  # You must define this function in Conditions_Thijs.py
from Determine_Correction_Factor_with_definitions import correction_factor_single_case

for casename in casenames:

    model, Re = parse_case_info(casename)

    transition = True
    cfd_results = data_cfd(model, Re, transition)

    # ic(cfd_results)

    print(model, Re)
    label = concise_label(casename)
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
        filename_unc_all = os.path.join(model, casename, 'unc_all_'+casename+'.txt')
        filename_unc = os.path.join(model, casename, 'unc_'+casename+'.txt')
        data_unc = load_and_preprocess_data_unc(filename_unc_all, filename_unc, model, STRIPS['CPwu_column'], STRIPS['CPwl_column'], PARAM)
        filename_corr = os.path.join(model, casename, 'corr_all_'+casename+'.txt')
        data_corr = load_and_preprocess_data_corr(filename_corr, model, STRIPS['CPwu_column'], STRIPS['CPwl_column'])
    elif model == 'V3':
        filename_unc_all = os.path.join(model, casename, 'unc_all_'+casename+'.txt')
        filename_unc = os.path.join(model, casename, 'unc_'+casename+'.txt')
        data_unc = load_and_preprocess_data_unc(filename_unc_all, filename_unc, model, STRIPS['CPwu_column'], STRIPS['CPwl_column'], PARAM)
        filename_corr = os.path.join(model, casename, 'corr_all_'+casename+'.txt')
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
    ## Only the first four columns are copied. These are [Alpha, Cd, Cl, Cm]
    POSTDATA_B_stripcorr = POSTDATA_A_presscorr[:, :4].copy()
    
    # Add 4 columns of zeros to POSTDATA_B_stripcorr (shape: same number of rows, 4 new columns)
    ## New columns added with [Alpha, Cd, Cl, Cm, Cl_low, Cl_mid, Cl_up, Cd_low, Cd_mid, Cd_up]
    POSTDATA_B_stripcorr = np.hstack([POSTDATA_B_stripcorr, np.zeros((POSTDATA_B_stripcorr.shape[0], 6))])

    POSTDATA_B_stripcorr[:, 2] = POSTDATA_B_stripcorr[:, 2] * PARAM['factor']

    # Copy B_stripcorr to C1_cdcorr
    POSTDATA_C1_cdcorr = POSTDATA_B_stripcorr.copy()

    ## # --- Extrapolate the Cd values when the wake rake cuts out
    # Create logical mask for "bad" values

    # alpha_kasper = [-10, -5, -2, 0, 2, 6, 10, 12, 14, 16, 18, 20, 22]
    # Cl_kasper = [-0.27, -0.152, -0.0115, 0.254, 0.59, 1.14, 1.53, 1.68, 1.73, 1.16, 1.08, 1.07, 1.07] ## V3
    # #Cl_kasper = [-0.28, -0.113, 0.0202, 0.323, 0.686, 1.24, 1.65, 1.81, 1.91, 1.52, 1.45, 1.39, 1.31] # Model2
    # Cd_kasper = [0.115, 0.071, 0.0505, 0.0366, 0.0275, 0.0161, 0.025, 0.0383, 0.0828, 0.185, 0.241, 0.287, 0.323]

    # Create interpolator for Cd_kasper
    ic(cfd_results['Alpha'])

    cd_interp = InterpolatedUnivariateSpline(cfd_results['Alpha'], cfd_results['Cd'], k=3, ext='extrapolate')

    # Find indices where alpha > PARAM['amax']
    extrap_idx = np.where(POSTDATA_C1_cdcorr[:, 0] > PARAM['amax'])[0]

    if extrap_idx.size > 0:
        # Interpolate Cd_kasper for these alpha values
        cd_extrapolated = cd_interp(POSTDATA_C1_cdcorr[extrap_idx, 0])
        # Replace values in POSTDATA_C1_cdcorr[:, 1] after PARAM['amax']
        POSTDATA_C1_cdcorr[extrap_idx, 1] = cd_extrapolated
    




    ## # --- Addd guestimate for moment
    ## Interpolate the data from Kasper's model. Then get the Cm per AoA in the dataframe in the fourth column.
    ## This can be used to correct the angle of attack in the future.
    # This will be done later

    ## Do make the first copy of the dataframe so code is future-proof
    POSTDATA_C2_cmcorr = POSTDATA_C1_cdcorr.copy()

    POSTDATA_C2_cmcorr[:,3] = get_cm_from_alpha(POSTDATA_C2_cmcorr[:,0], model, Re)

    POSTDATA_final = POSTDATA_C2_cmcorr.copy()
    PLACEHOLDER_low = 0 ## This should be the array with interpolated values per angle of attack of the Cd by kasper

    PLACEHOLDER_up = cd_interp(POSTDATA_final[:, 0]) * 2
    PLACEHOLDER_mid = cd_interp(POSTDATA_final[:, 0]) 
    ## Alpha correction
    POSTDATA_final[:,0] = POSTDATA_final[:,0] + 57.3 * PARAM['sigma'] / (PARAM['beta'] * 2 * np.pi) * (POSTDATA_final[:,2] + POSTDATA_final[:,3])

    ## Correction parameters
    PARAM['eps_solid_blockage'] = (2 - PARAM['M'] ** 2) * PARAM['factor_sb'] * (1 + PARAM['eps_alpha'] * (POSTDATA_final[:, 0] * np.pi / 180) ** 2)
    PARAM['eps_wake_blockage_measured'] = (2 - PARAM['M'] ** 2) * (PARAM['factor_wb']*POSTDATA_final[:,1]) ## Using the measured drag coefficient
    PARAM['eps_wake_blockage_low'] = (2 - PARAM['M'] ** 2) * (PARAM['factor_wb']*PLACEHOLDER_low) ## Using the lower bound drag coeffcient
    PARAM['eps_wake_blockage_mid'] = (2 - PARAM['M'] ** 2) * (PARAM['factor_wb']*PLACEHOLDER_mid) ## Using the lower bound drag coeffcient
    PARAM['eps_wake_blockage_up'] = (2 - PARAM['M'] ** 2) * (PARAM['factor_wb']*PLACEHOLDER_up) ## Using the upper bound drag coeffcient

    ## Cl correction --> Measured, Lower, Upper
    POSTDATA_final[:,2] = POSTDATA_final[:,2] * (1 - PARAM['factor_sc'] + 5.25 * PARAM['factor_sc']**2 - PARAM['eps_solid_blockage'] - PARAM['eps_wake_blockage_measured'])
    POSTDATA_final[:,4] = POSTDATA_final[:,2] * (1 - PARAM['factor_sc'] + 5.25 * PARAM['factor_sc']**2 - PARAM['eps_solid_blockage'] - PARAM['eps_wake_blockage_low'])
    POSTDATA_final[:,5] = POSTDATA_final[:,2] * (1 - PARAM['factor_sc'] + 5.25 * PARAM['factor_sc']**2 - PARAM['eps_solid_blockage'] - PARAM['eps_wake_blockage_mid'])
    POSTDATA_final[:,6] = POSTDATA_final[:,2] * (1 - PARAM['factor_sc'] + 5.25 * PARAM['factor_sc']**2 - PARAM['eps_solid_blockage'] - PARAM['eps_wake_blockage_up'])

    ## Cd correction --> Measured, Lower, Upper
    POSTDATA_final[:,1] = POSTDATA_final[:,1] * (1 - PARAM['eps_solid_blockage'] - PARAM['eps_wake_blockage_measured'])
    POSTDATA_final[:,7] = POSTDATA_final[:,1] * (1 - PARAM['eps_solid_blockage'] - PARAM['eps_wake_blockage_low'])
    POSTDATA_final[:,8] = POSTDATA_final[:,1] * (1 - PARAM['eps_solid_blockage'] - PARAM['eps_wake_blockage_mid'])
    POSTDATA_final[:,9] = POSTDATA_final[:,1] * (1 - PARAM['eps_solid_blockage'] - PARAM['eps_wake_blockage_up'])

    # Convert POSTDATA_final to a DataFrame (if not already)
    POSTDATA_final_df = pd.DataFrame(POSTDATA_final, columns=[
        'alpha', 'Cd', 'Cl', 'Cm', 'Cl_low', 'Cl_mid', 'Cl_up', 'Cd_low', 'Cd_mid', 'Cd_up'
    ])

    POSTDATA_final_df['alpha_unc'] = POSTDATA_A_presscorr[:,0].round(1)  # Round alpha to 1 decimal place

    # Group by angle of attack (alpha) and calculate mean and std
    POSTDATA_final_df['alpha_unc'] = POSTDATA_final_df['alpha_unc'].round(1)  # Round alpha to 1 decimal place

    grouped = POSTDATA_final_df.groupby('alpha_unc')
    mean_df = grouped.mean()
    POSTDATA_final = mean_df
    std_df = grouped.std()

    # Find the index where alpha reaches or exceeds PARAM['amax']
    amax_idx = np.argmax(POSTDATA_final['alpha'].values >= PARAM['amax'])

    # If PARAM['amax'] is not found, plot all as normal
    if POSTDATA_final['alpha'].iloc[amax_idx] < PARAM['amax']:
        amax_idx = len(POSTDATA_final)


    POSTDATA_final = POSTDATA_final.values

    # Determine start index for plotting based on PARAM['skip_idx']
    start_idx = 1 if PARAM['skip_idx'][0] == 0 else 0

    # ic(PARAM['skip_idx'])
    # Plot measured Cl (up to amax_idx) in one color, skipping first row if needed
    ax1.plot(
        POSTDATA_final[start_idx:amax_idx+1, 0],
        POSTDATA_final[start_idx:amax_idx+1, 2],
        '-x', linewidth=2, color='C0', label='WT data'
    )

    # Plot measured Cd (up to amax_idx) in one color, skipping first row if needed
    ax2.plot(
        POSTDATA_final[start_idx:amax_idx+1, 0],
        POSTDATA_final[start_idx:amax_idx+1, 1],
        '-x', linewidth=2, color='C0', label='WT data'
    )

    # Plot extrapolated Cl (from amax_idx onward) in another color, starting from the last measured point
    if amax_idx < len(POSTDATA_final) - 1:
        ax1.plot(POSTDATA_final[amax_idx:, 0], POSTDATA_final[amax_idx:, 5], '-x', linewidth=2, color='red', label=f'Extrapolated Cl data')

    # Plot extrapolated Cd (from amax_idx onward) in another color, starting from the last measured point
    # if amax_idx < len(POSTDATA_final) - 1:
    #     ax2.plot(POSTDATA_final[amax_idx:, 0], POSTDATA_final[amax_idx:, 8], '-x', linewidth=2, color='red', label=f'Extrapolated')

    # Plot shaded area for the remaining values (if any)

        # # For drag coefficient (Cd)
        # ax2.fill_between(
        #     POSTDATA_final[amax_idx:, 0],
        #     POSTDATA_final[amax_idx:, 7],  # lower bound
        #     POSTDATA_final[amax_idx:, 9],  # upper bound
        #     color='red', alpha=0.3, label=f'Extrapolated uncertainty'
        # )

    # Convert std_df to numpy array for easy indexing
    std_arr = std_df.values

    # Plot confidence interval for Cl (measured region)
    ax1.fill_between(
        POSTDATA_final[start_idx:amax_idx+1, 0],  # alpha
        POSTDATA_final[start_idx:amax_idx+1, 2] - 3 * std_arr[start_idx:amax_idx+1, 2],  # mean - std
        POSTDATA_final[start_idx:amax_idx+1, 2] + 3 * std_arr[start_idx:amax_idx+1, 2],  # mean + std
        color='blue', alpha=0.3, label=f'CI of 99%'
    )

    # Plot confidence interval for Cd (measured region)
    ax2.fill_between(
        POSTDATA_final[start_idx:amax_idx+1, 0],  # alpha
        POSTDATA_final[start_idx:amax_idx+1, 1] - 3 * std_arr[start_idx:amax_idx+1, 1],  # mean - std
        POSTDATA_final[start_idx:amax_idx+1, 1] + 3 * std_arr[start_idx:amax_idx+1, 1],  # mean + std
        color='blue', alpha=0.3, label=f'CI of 99%'
    )

    if amax_idx < len(POSTDATA_final):
    # For lift coefficient (Cl)
        ax1.fill_between(
            POSTDATA_final[amax_idx:, 0],
            POSTDATA_final[amax_idx:, 4],  # lower bound
            POSTDATA_final[amax_idx:, 6],  # upper bound
            color='red', alpha=0.3, label=f'Extrapolated uncertainty'
        )

ax1.plot(cfd_results['Alpha'], cfd_results['Cl'], '--o', linewidth=2, color='g', label='CFD data')
ax1.set_xlabel(r'$\alpha$ ($^\circ$)', fontsize=18)
ax1.set_ylabel(r'$C_\mathrm{l}$ (-)', fontsize=18)
# ax1.set_title('Lift polar compared to literature', fontsize=18)


# ax2.plot(POSTDATA_final[:,0], POSTDATA_C1_cdcorr[:,1], label='full exprapolate')
ax2.plot(cfd_results['Alpha'], cfd_results['Cd'], '--o', linewidth=2, color='g', label='CFD data')
ax2.set_xlabel(r'$\alpha$ ($^\circ$)', fontsize=18)
ax2.set_ylabel(r'$C_\mathrm{d}$ (-)', fontsize=18)
# ax2.set_title('Drag polar compared to literature', fontsize=18)


for ax in [ax1, ax2]:
    ax.set_xlim(-10, 25)
    ax.set_xticks(np.arange(-10, 26, 5))

# Example: Set y-axis limits (adjust as needed for your data)
ax1.set_ylim(-0.5, 2.0)   # Cl axis
ax1.set_yticks(np.arange(-0.5, 2.6, 0.5))

ax2.set_ylim(0, 0.4)      # Cd axis
ax2.set_yticks(np.arange(0, 0.41, 0.1))

handles, labels = ax1.get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.0),  # Move legend bar up into the figure
    ncol=len(labels),
    frameon=False,
    fontsize=18
)



# fig.savefig(r'C:\TU_Delft\Master\Thesis\Figures overleaf\Results\CFD_comp\V3_Re1e6_transition_polars.png', dpi=300, bbox_inches='tight')


plt.show()
# ## # --- VALUES FROM KASKPERS MODEL ---
# ic(pd.DataFrame(POSTDATA_final))
# ic(cfd_results)



# ## # --- Plotting the results ---
# fig = plt.figure(figsize=(12, 4))  # Similar to [100 100 1200 360] in pixels

# # First subplot: Lift polar
# ax1 = fig.add_axes([0.08, 0.2, 0.4, 0.72])
# ax1.grid(True)
# ax1.tick_params(labelsize=18)
# ax1.plot(data_unc.values[:, 0], data_unc.values[:, 2], '-x', linewidth=2, color=CC[0], label='unc')
# ax1.plot(POSTDATA_D1_wallcorr[:, 0], POSTDATA_D1_wallcorr[:, 2], '-x', linewidth=2, color=CC[1], label='corr')
# ax1.plot(POSTDATA_D2_wallcdcmcorr[:, 0], POSTDATA_D2_wallcdcmcorr[:, 2], '-x', linewidth=2, color=CC[2], label='corr extrap')
# ax1.plot(data_corr.values[:, 0], data_corr.values[:, 2], '--', linewidth=2, color='k', label='old corr - ref')
# ax1.plot(alpha_kasper, Cl_kasper, '--', linewidth=2, color='g', label='Kasper model')
# ax1.set_xlabel(r'Angle of attack, $\alpha$ [deg]', fontsize=18)
# ax1.set_ylabel(r'Lift coefficient, $C_l$ [-]', fontsize=18)
# ax1.set_title(f'Lift polar - {casename1}', fontsize=18)

# # Second subplot: Drag polar
# ax2 = fig.add_axes([0.58, 0.2, 0.4, 0.72])
# ax2.grid(True)
# ax2.tick_params(labelsize=18)
# ax2.plot(data_unc.values[:, 0], data_unc.values[:, 1], '-x', linewidth=2, color=CC[0], label='unc')
# ax2.plot(POSTDATA_D1_wallcorr[:, 0], POSTDATA_D1_wallcorr[:, 1], '-x', linewidth=2, color=CC[1], label='corr')
# ax2.plot(POSTDATA_D2_wallcdcmcorr[:, 0], POSTDATA_D2_wallcdcmcorr[:, 1], '-x', linewidth=2, color=CC[2], label='corr extrap')
# ax2.plot(data_corr.values[:, 0], data_corr.values[:, 1], '--', linewidth=2, color='k', label='old corr - ref')
# # ax2.plot(alpha_kasper, Cd_kasper, '--', linewidth=2, color='g', label='Kasper model')
# ax2.set_xlabel(r'Angle of attack, $\alpha$ [deg]', fontsize=18)
# ax2.set_ylabel(r'Drag coefficient, $C_d$ [-]', fontsize=18)
# ax2.set_title(f'Drag polar - {casename1}', fontsize=18)
# ax2.legend(loc='upper left', fontsize=14)

# plt.show()

# # ic(POSTDATA_D2_wallcdcmcorr[:, 0])  # Print the angle of attack values
# # ic(POSTDATA_D2_wallcdcmcorr[:, 2])  # Print the lift coefficient values

