import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import trapz
from icecream import ic
import os

def load_and_preprocess_data(filename, model, z_min=340, z_max=380, alpha_col=1, tol=0.05):
    data = pd.read_csv(filename, sep='\t', header=None, skiprows=1, usecols=range(145))
    first_row = data.iloc[[0]].copy()
    rest_rows = data.iloc[1:].copy()
    rest_rows = rest_rows.apply(pd.to_numeric, errors='coerce')
    if model == 'Model2':
        rest_rows = rest_rows[(rest_rows[9] >= z_min) & (rest_rows[9] <= z_max)]
        rest_rows['alpha_bin'] = rest_rows[alpha_col].round(1)
        grouped = rest_rows.groupby('alpha_bin', as_index=False, dropna=False).mean()
        grouped = grouped.drop(columns=['alpha_bin'])
        data = pd.concat([first_row, grouped], ignore_index=True)
    elif model == 'V3':
        data = pd.concat([first_row, rest_rows], ignore_index=True)
    return data

def extract_pressure_positions(data, CPwu_column, CPwl_column):
    pressposCpwu = data.iloc[0, CPwu_column].to_numpy(dtype=float)
    pressposCpwl = data.iloc[0, CPwl_column].to_numpy(dtype=float)
    ic(pressposCpwu)
    ic(pressposCpwl)
    return pressposCpwu, pressposCpwl

def extract_measurements(data, alpha_column, CPwu_column, CPwl_column):
    alpha = data.iloc[1:, alpha_column].to_numpy(dtype=float)
    Cpwu = data.iloc[1:, CPwu_column].to_numpy(dtype=float)
    Cpwl = data.iloc[1:, CPwl_column].to_numpy(dtype=float)
    diffCp = Cpwl - Cpwu
    return alpha, Cpwu, Cpwl, diffCp

def calculate_correction_factor(alpha, diffCp, presspos, alpha_min, alpha_max, method='Fmincon'):
    m = alpha.shape[0]
    rangealpha = np.where((alpha >= alpha_min) & (alpha <= alpha_max))[0]
    CpmaxT = np.zeros(m)
    for i in range(m):
        if np.max(diffCp[i, :]) > 0:
            CpmaxT[i] = np.max(diffCp[i, :])
        else:
            CpmaxT[i] = -np.max(np.abs(diffCp[i, :]))
    OPT = np.zeros((m, 3))
    error = np.zeros(m)
    for i in rangealpha:
        if method == 'Fmincon':
            def func(x):
                model = x[2] / np.cosh((np.pi / x[0]) * (presspos / 1000 - x[1]))
                return np.sum((model - diffCp[i, :]) ** 2)
            res = minimize(func, [1.6, 1.0, CpmaxT[i]], method='SLSQP')
            OPT[i, :] = res.x
            error[i] = res.fun
        else:
            raise ValueError("Invalid method selected. Choose 'Fmincon'.")
    x = np.arange(-10, 10.05, 0.05)
    Cpth = np.array([
        opt[2] / np.cosh((np.pi / opt[0]) * (x - opt[1])) if np.all(opt) else np.full_like(x, np.nan)
        for opt in OPT
    ])
    factor = np.full(m, np.nan)
    for i in rangealpha:
        Cl_theory = trapz(Cpth[i, :], x)
        Cl_meas = trapz(diffCp[i, :], presspos / 1000)
        factor[i] = Cl_theory / Cl_meas if Cl_meas != 0 else np.nan
    return factor, rangealpha, x, Cpth

def plot_correction_factors(alphas_list, factors_list, labels):
    plt.figure(figsize=(8, 5))
    for alpha, factor, label in zip(alphas_list, factors_list, labels):
        plt.plot(alpha, factor, 'o', label=label)
    plt.xlabel('Angle of attack (deg)')
    plt.ylabel('Correction factor')
    plt.ylim([-0.5, 3])
    plt.title('Correction factor vs. Angle of attack')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_fitting_subplots(presspos, diffCp, x, Cpth, alpha, rangealpha):
    """
    Plots measured and calculated Cp for each angle of attack in subplots.

    Parameters:
        presspos (array): Pressure positions.
        diffCp (2D array): Measured Cp differences.
        x (array): X positions for calculated Cp.
        Cpth (2D array): Calculated Cp values.
        alpha (array): Angles of attack.
        rangealpha (array): Indices of angles to plot.
    """
    p = len(rangealpha)
    plt.figure(figsize=(12, 6))
    for idx, i in enumerate(rangealpha):
        plt.subplot(2, int(np.ceil(p / 2)), idx + 1)
        plt.plot(presspos, diffCp[i, :], 'x', label='Measured')
        plt.plot(1000 * x, Cpth[i, :], '-', label='Calculated')
        plt.xlabel('X')
        plt.ylabel('Cp')
        plt.title(f'Angle = {alpha[i]:.1f}°')
        plt.xlim([-2.5e3, 5e3])
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    plt.show()

# ==== Main script logic ====
Final_Factor = True # Set to True if you want to calculate the final correction factor for a specific case
All_Factors = False # Set to True if you want to calculate all correction factors for every Angle of Attack
Provide_plot = True # Set to True if you want to plot the fitting subplots
alpha_min = -10 if All_Factors else 2
alpha_max = 25 if All_Factors else 18


model = 'V3' ## Specify model as additional preprocessing step has to be done for model2
if model == 'Model2':
    casenames = ['Model2_no_zz_Re_1e6', 'Model2_no_zz_Re_2e6', 'Model2_no_zz_Re_5e5',
                 'Model2_small_zz_bottom_Re_1e6', 'Model2_small_zz_bottom_Re_5e5',
                 'Model2_zz_0.1c_top_Re_1e6', 'Model2_zz_0.1c_top_Re_5e5',
                 'Model2_zz_0.05c_top_Re_1e6', 'Model2_zz_0.05c_top_Re_5e5',
                 'Model2_zz_bottom_0.05c_top_Re_1e6', 'Model2_zz_bottom_0.05c_top_Re_5e5',
                 'Model2_zz_bottom_Re_1e6', 'Model2_zz_bottom_Re_5e5'] ## State all different cases here
    fit_case = 'Model2_no_zz_Re_5e5'  # Specify the case to use for fitting plot
elif model == 'V3':
    casenames = ['V3_no_zz_Re_1e6', 'V3_no_zz_Re_15e5', 'V3_no_zz_Re_5e5',
                'V3_small_zz_bottom_Re_1e6', 'V3_small_zz_bottom_Re_5e5',
                'V3_zz_0.05c_top_Re_1e6', 'V3_zz_0.05c_top_Re_5e5',
                'V3_zz_bottom_0.05c_top_Re_1e6', 'V3_zz_bottom_0.05c_top_Re_5e5',
                'V3_bottom_0.03c_top_Re_1e6', 'V3_bottom_45deg_0.03c_top_Re_1e6', 
                'V3_bottom_45_deg_Re_1e6','V3_bottom_45_deg_Re_5e5']
    fit_case = 'V3_bottom_45_deg_Re_1e6'


## Locations of certain parameters
CPwu_column = list(range(101, 123))
CPwl_column = list(range(123, 145))
alpha_column = 1
Method = 'Fmincon'

alphas_list = []
factors_list = []
labels = []

## Generate plot of correction factor per angle of attack for all cases
if All_Factors == True:
    for casename in casenames:
        if model == 'Model2':
            filename = os.path.join(model, casename, 'unc_all_'+casename+'.txt')
            data = load_and_preprocess_data(filename, model)
        elif model == 'V3':
            filename = os.path.join(model, casename, 'unc_'+casename+'.txt')
            data = load_and_preprocess_data(filename, model)
        pressposCpwu, pressposCpwl = extract_pressure_positions(data, CPwu_column, CPwl_column)
        presspos = pressposCpwu
        alpha, Cpwu, Cpwl, diffCp = extract_measurements(data, alpha_column, CPwu_column, CPwl_column)
        factor, rangealpha, x, Cpth = calculate_correction_factor(alpha, diffCp, presspos, alpha_min, alpha_max, Method)
        alphas_list.append(alpha)
        factors_list.append(factor)
        labels.append(casename)

    # Plot all correction factors for all cases
    plot_correction_factors(alphas_list, factors_list, labels)
    exit()

## Provide factor for a specific test case including plotting if selected
if Final_Factor and fit_case in casenames:
    # Only process the specified fit_case
    casename = fit_case
    if model == 'Model2':
        filename = os.path.join(model, casename, 'unc_all_'+casename+'.txt')
        data = load_and_preprocess_data(filename, model)
    elif model == 'V3':
        filename = os.path.join(model, casename, 'unc_'+casename+'.txt')
        data = load_and_preprocess_data(filename, model)
    pressposCpwu, pressposCpwl = extract_pressure_positions(data, CPwu_column, CPwl_column)
    presspos = pressposCpwu
    ic(presspos)
    alpha, Cpwu, Cpwl, diffCp = extract_measurements(data, alpha_column, CPwu_column, CPwl_column)
    factor, rangealpha, x, Cpth = calculate_correction_factor(alpha, diffCp, presspos, alpha_min, alpha_max, Method)

    # Fitting plot
    if Provide_plot:
        plot_fitting_subplots(presspos, diffCp, x, Cpth, alpha, rangealpha)
    print(np.nanmean(factor))

##Provide factor for every test case without plotting
if Final_Factor == False:
    for casename in casenames:
        if model == 'Model2':
            filename = os.path.join(model, casename, 'unc_all_'+casename+'.txt')
            data = load_and_preprocess_data(filename, model)
        elif model == 'V3':
            filename = os.path.join(model, casename, 'unc_'+casename+'.txt')
            data = load_and_preprocess_data(filename, model)
        pressposCpwu, pressposCpwl = extract_pressure_positions(data, CPwu_column, CPwl_column)
        presspos = pressposCpwu
        alpha, Cpwu, Cpwl, diffCp = extract_measurements(data, alpha_column, CPwu_column, CPwl_column)
        factor, rangealpha, x, Cpth = calculate_correction_factor(alpha, diffCp, presspos, alpha_min, alpha_max, Method)
        print(np.nanmean(factor), ' for case ' ,casename)

