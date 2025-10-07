import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import trapezoid
from icecream import ic
import os

from plot_styling import set_plot_style

set_plot_style()

def load_and_preprocess_data(
    filename,
    model,
    CPwu_column,
    CPwl_column,
    PARAM,
    z_min=340,
    z_max=380,
    alpha_col=1,
    tol=0.05,
):
    # Skip the first two rows
    data = pd.read_csv(filename, sep="\t", header=None, skiprows=2, usecols=range(145))
    # Convert all data to numeric, set errors to NaN
    data = data.apply(pd.to_numeric, errors="coerce")

    # Store average Mach number (13th column, index 12) -- - average used in future
    PARAM["M"] = data[12].mean()
    if model == "Model2":
        data = data[((data[9] >= z_min) & (data[9] <= z_max)) | (data[9] == 260)]
        data["alpha_bin"] = data[alpha_col].round(1)
        grouped = data.groupby("alpha_bin", as_index=False, dropna=False).mean()
        grouped = grouped.drop(columns=["alpha_bin"])
        data = grouped
        # Find first occurrence where z == 260 and store corresponding alpha in PARAM['amax']. This is where the wake rake cuts out.
        idx = data[data[9] == 260].index
        if not idx.empty:
            PARAM["amax"] = data.iloc[idx[0], alpha_col]
        else:
            PARAM["amax"] = 50  # Handle case where no z == 260 is found
    elif model == "V3":

        # For V3, just use the data as is (already numeric)
        # Find first occurrence where z == 40 and store corresponding alpha in PARAM['amax']. This is where the wake rake cuts out.
        idx = data[data[9] == 40].index
        if not idx.empty:
            PARAM["amax"] = data.iloc[idx[0], alpha_col]
        else:
            PARAM["amax"] = 50  # Handle case where no z == 40 is found
    # Columns to keep: 1, 5, 6, 7, and all in CPwu_column and CPwl_column
    keep_cols = [1, 5, 6, 7] + CPwu_column + CPwl_column
    keep_cols = sorted(set(keep_cols))
    data = data.iloc[:, keep_cols]
    return data


def extract_measurements(data, STRIPS):
    n_base = 4
    n_cpwu = len(STRIPS["CPwu_column"])
    n_cpwl = len(STRIPS["CPwl_column"])
    alpha = data.iloc[0:, 0].to_numpy(
        dtype=float
    )  # First column after reduction is alpha
    Cpwu = data.iloc[0:, n_base : n_base + n_cpwu].to_numpy(dtype=float)
    Cpwl = data.iloc[0:, n_base + n_cpwu : n_base + n_cpwu + n_cpwl].to_numpy(
        dtype=float
    )
    diffCp = Cpwl - Cpwu
    return alpha, Cpwu, Cpwl, diffCp


def calculate_correction_factor(
    alpha, diffCp, presspos, alpha_min, alpha_max, method="Fmincon"
):
    m = alpha.shape[0]
    rangealpha = np.where((alpha >= alpha_min) & (alpha <= alpha_max))[0]
    CpmaxT = np.zeros(m)
    for i in range(m):
        # ic(pd.DataFrame(diffCp))
        if np.max(diffCp[i, :]) > 0:
            CpmaxT[i] = np.max(diffCp[i, :])
        else:
            CpmaxT[i] = -np.max(np.abs(diffCp[i, :]))
    OPT = np.zeros((m, 3))
    error = np.zeros(m)
    for i in rangealpha:
        if method == "Fmincon":

            def func(x):
                model = x[2] / np.cosh((np.pi / x[0]) * (presspos - x[1]))
                return np.sum((model - diffCp[i, :]) ** 2)

            res = minimize(func, [1.6, 1.0, CpmaxT[i]], method="SLSQP")
            OPT[i, :] = res.x
            # ic(OPT)
            error[i] = res.fun
        else:
            raise ValueError("Invalid method selected. Choose 'Fmincon'.")
    x = np.arange(-10, 10.05, 0.05)
    Cpth = np.array(
        [
            (
                opt[2] / np.cosh((np.pi / opt[0]) * (x - opt[1]))
                if np.all(opt)
                else np.full_like(x, np.nan)
            )
            for opt in OPT
        ]
    )
    factor = np.full(m, np.nan)
    for i in rangealpha:
        Cl_theory = trapezoid(Cpth[i, :], x)
        Cl_meas = trapezoid(diffCp[i, :], presspos)
        factor[i] = Cl_theory / Cl_meas if Cl_meas != 0 else np.nan
    return factor, rangealpha, x, Cpth


def plot_correction_factors(model, alphas_list, factors_list, labels):
    fig, ax = plt.subplots(figsize=(10, 5))
    for alpha, factor, label in zip(alphas_list, factors_list, labels):
        ax.plot(alpha, factor, "o", label=label)

    ax.set_xlabel(r"$\alpha$ ($^\circ$)")
    ax.set_ylabel(r"$\eta$ (-)")
    ax.set_ylim([1, 1.5])
    ax.set_xlim(-10, 25)
    ax.set_xticks(range(-10, 26, 5))
    ax.grid(True)

    # Put legend outside the figure (bottom), no frame, 3 columns
    handles, lbls = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        lbls,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),  # move legend below figure
        ncol=3,
        frameon=False,
    )

    # Make room for the outside legend and save
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.18)  # increase bottom margin so legend is visible
    fig.savefig(f"results/all_correction_factors_{model}.pdf", bbox_inches="tight")
    plt.show()


def plot_fitting_subplots(presspos, diffCp, x, Cpth, alpha, rangealpha):
    p = len(rangealpha)
    plt.figure(figsize=(10, 4))
    for idx, i in enumerate(rangealpha):
        # plt.subplot(2, int(np.ceil(p / 2)), idx + 1)
        plt.plot(presspos, diffCp[i, :], "x", label="Measured")
        plt.plot(x, Cpth[i, :], "-", label="Calculated")
        plt.xlabel("x (m)")
        plt.ylabel(r"$Cp$ (-)")
        # plt.title(f"Angle = {alpha[i]:.1f}°")
        plt.xlim([-2.5, 5])
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    plt.savefig("results/Fitting_subplots.pdf")
    plt.show()


def correction_factors_all_cases(
    casenames, model, STRIPS, alpha_min, alpha_max, Method
):
    alphas_list = []
    factors_list = []
    labels = []
    for casename in casenames:
        PARAM = {}
        if model == "Model2":
            filename = os.path.join(model, casename, "unc_all_" + casename + ".txt")
        elif model == "V3":
            filename = os.path.join(model, casename, "unc_" + casename + ".txt")
        data = load_and_preprocess_data(
            filename, model, STRIPS["CPwu_column"], STRIPS["CPwl_column"], PARAM
        )
        alpha, Cpwu, Cpwl, diffCp = extract_measurements(data, STRIPS)
        factor, rangealpha, x, Cpth = calculate_correction_factor(
            alpha, diffCp, STRIPS["x"], alpha_min, alpha_max, Method
        )
        alphas_list.append(alpha)
        factors_list.append(factor)
        # Remove 'small_' from label if present
        label = casename.replace("small_", "")
        label = label.replace("V3_bottom", "V3_zz_bottom")
        label = label.replace("Model2_", "").replace("V3_", "")

        labels.append(label)
    plot_correction_factors(model, alphas_list, factors_list, labels)
    return alphas_list, factors_list, labels


def correction_factor_single_case(
    casename, model, STRIPS, alpha_min, alpha_max, Method, provide_plot=True
):
    PARAM = {}
    if model == "Model2":
        filename = os.path.join(model, casename, "unc_all_" + casename + ".txt")
    elif model == "V3":
        filename = os.path.join(model, casename, "unc_" + casename + ".txt")

    data = load_and_preprocess_data(
        filename, model, STRIPS["CPwu_column"], STRIPS["CPwl_column"], PARAM
    )
    alpha, Cpwu, Cpwl, diffCp = extract_measurements(data, STRIPS)
    # ic(diffCp)
    factor, rangealpha, x, Cpth = calculate_correction_factor(
        alpha, diffCp, STRIPS["x"], alpha_min, alpha_max, Method
    )
    if provide_plot:
        plot_fitting_subplots(STRIPS["x"], diffCp, x, Cpth, alpha, rangealpha)
    print(np.nanmean(factor), " for ", casename)
    return np.nanmean(factor)  # , factor, alpha, diffCp


def correction_factors_no_plot(
    casenames,
    model,
    STRIPS,
    alpha_min,
    alpha_max,
    Method,
):
    results = {}
    for casename in casenames:
        PARAM = {}
        if model == "Model2":
            filename = os.path.join(model, casename, "unc_all_" + casename + ".txt")
        elif model == "V3":
            filename = os.path.join(model, casename, "unc_" + casename + ".txt")

        data = load_and_preprocess_data(
            filename, model, STRIPS["CPwu_column"], STRIPS["CPwl_column"], PARAM
        )
        alpha, Cpwu, Cpwl, diffCp = extract_measurements(data, STRIPS)
        factor, rangealpha, x, Cpth = calculate_correction_factor(
            alpha, diffCp, STRIPS["x"], alpha_min, alpha_max, Method
        )
        print(np.nanmean(factor), " for case ", casename)
        results[casename] = np.nanmean(factor)
    return results


def run_main():
    ## # --- CHANGE PARAMETERS FROM HERE ---

    model = "V3"  ## Specify model as additional preprocessing step has to be done for model2
    if model == "Model2":
        casenames = [
            "Model2_no_zz_Re_5e5",
            "Model2_no_zz_Re_1e6",
            "Model2_no_zz_Re_2e6",
            "Model2_small_zz_bottom_Re_5e5",
            "Model2_small_zz_bottom_Re_1e6",
            "Model2_zz_0.1c_top_Re_5e5",
            "Model2_zz_0.1c_top_Re_1e6",
            "Model2_zz_0.05c_top_Re_5e5",
            "Model2_zz_0.05c_top_Re_1e6",
            "Model2_zz_bottom_0.05c_top_Re_5e5",
            "Model2_zz_bottom_0.05c_top_Re_1e6",
        ]  # ,
        # 'Model2_zz_bottom_Re_1e6', 'Model2_zz_bottom_Re_5e5'] ## State all different cases here
        fit_case = "Model2_zz_bottom_Re_5e5"  # Specify the case to use for fitting plot
    elif model == "V3":
        casenames = [
            "V3_no_zz_Re_5e5",
            "V3_no_zz_Re_1e6",
            "V3_no_zz_Re_15e5",
            "V3_small_zz_bottom_Re_5e5",
            "V3_small_zz_bottom_Re_1e6",
            "V3_zz_0.05c_top_Re_5e5",
            "V3_zz_0.05c_top_Re_1e6",
            "V3_zz_bottom_0.05c_top_Re_5e5",
            "V3_zz_bottom_0.05c_top_Re_1e6",
            #'V3_bottom_0.03c_top_Re_1e6',
            "V3_bottom_45_deg_Re_5e5",
            "V3_bottom_45_deg_Re_1e6",
            "V3_bottom_45deg_0.03c_top_Re_1e6",
        ]
        fit_case = "V3_no_zz_Re_1e6"

    # --- STRIPS dictionary for pressure tab locations ---
    STRIPS = {}
    STRIPS["CPwu_column"] = list(range(101, 123))
    STRIPS["CPwl_column"] = list(range(123, 145))
    STRIPS["x"] = (
        np.array(
            [
                0,
                178,
                333,
                468,
                585,
                685,
                775,
                851,
                918,
                975,
                1025,
                1070,
                1120,
                1177,
                1244,
                1320,
                1410,
                1510,
                1627,
                1762,
                1917,
                2095,
            ]
        )
        / 1000
    )

    Method = "Fmincon"

    n_base = 4  # Number of base columns: [1, 5, 6, 7]
    n_cpwu = len(STRIPS["CPwu_column"])
    n_cpwl = len(STRIPS["CPwl_column"])

    alphas_list = []
    factors_list = []
    labels = []

    a_min_fullrange = -10
    a_max_fullrange = 25

    # alpha_min = 2
    # alpha_max = 8

    #To show angle of attack = 8 
    alpha_min = 7.9
    alpha_max = 9
    ## # --- Run the functions ---
    correction_factors_all_cases(
        casenames, model, STRIPS, a_min_fullrange, a_max_fullrange, Method
    )
    # correction_factor_single_case(fit_case, model, STRIPS, alpha_min, alpha_max, Method, provide_plot=True)
    # results = correction_factors_no_plot(casenames, model, STRIPS, alpha_min, alpha_max, Method)
    return


if __name__ == "__main__":
    run_main()
