"""Process wind-tunnel and CFD data into combined datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import scipy.integrate as integrate
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d

if not hasattr(integrate, "trapezoid"):
    integrate.trapezoid = np.trapezoid

from Determine_Correction_Factor_with_definitions import (
    correction_factor_single_case,
)
from utils import PROCESSED_DATA_DIR, PROJECT_DIR, ensure_directory

DEFAULT_CASES = ["V3_no_zz_Re_1e6"]
DEFAULT_METHOD = "Fmincon"
DEFAULT_ALPHA_MIN = 2.0
DEFAULT_ALPHA_MAX = 8.0

STRIPS = {
    "CPwu_column": list(range(101, 123)),
    "CPwl_column": list(range(123, 145)),
    "x": np.array(
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
    / 1000,
}


def parse_case_info(case_string: str) -> Tuple[str, str]:
    parts = case_string.split("_")
    model = parts[0]
    Re = None
    for part in parts:
        if part.startswith("Re"):
            Re = part.split("Re")[1]
            if not Re:
                idx = parts.index(part)
                Re = parts[idx + 1]
            break
    if Re is None:
        raise ValueError(f"Could not parse Reynolds number from case '{case_string}'.")
    return model, Re


def get_model_parameters(model: str) -> Dict[str, float]:
    if model == "Model2":
        return {
            "t_c": 0.0078,
            "lambda": 0.0341,
            "c": 0.5,
            "c1": 0.5,
            "h": 1.656,
            "amin": -50,
        }
    if model == "V3":
        return {
            "t_c": 0.1066,
            "lambda": 0.0539,
            "c": 0.51948,
            "c1": 0.51948,
            "h": 1.656,
            "amin": -50,
        }
    raise ValueError(f"Unsupported model '{model}'.")


def data_cfd(model: str, Re: str, transition: bool) -> pd.DataFrame:
    root_path = PROJECT_DIR / "CFD results"
    model_folder = model if model in {"V3", "Model2"} else model.capitalize()
    model_path = root_path / model_folder
    if not model_path.exists():
        raise FileNotFoundError(f"Missing CFD model folder: {model_path}")

    if Re == "1e6":
        data_file = "Re1e6 transition" if transition else "Re1e6 fully turbulent"
    elif Re == "5e5":
        data_file = "Re5e5 transition" if transition else "Re5e5 fully turbulent"
    else:
        raise ValueError("Re must be '1e6' or '5e5'.")

    file_path = model_path / data_file / "polar.dat"
    if not file_path.exists():
        raise FileNotFoundError(f"Could not locate CFD data '{file_path}'.")

    with file_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split()
    df = pd.read_csv(file_path, sep=r"\s+", header=None, skiprows=1, names=header)
    if "iter_n" in df.columns:
        df = df[df["iter_n"] != 5001]
    return df


def load_and_preprocess_data_unc(
    filename_unc_all: Path,
    filename_unc: Path,
    model: str,
    CPwu_column: List[int],
    CPwl_column: List[int],
    params: Dict[str, float],
    z_min: int = 340,
    z_max: int = 380,
    alpha_col: int = 1,
    tol: float = 0.05,
) -> pd.DataFrame:
    data_unc_all = pd.read_csv(
        filename_unc_all,
        sep="\t",
        header=None,
        skiprows=2,
        usecols=range(145),
    )
    data_unc = pd.read_csv(
        filename_unc,
        sep="\t",
        header=None,
        skiprows=2,
        usecols=range(145),
    )
    data_unc_all = data_unc_all.apply(pd.to_numeric, errors="coerce")
    data_unc = data_unc.apply(pd.to_numeric, errors="coerce")
    params["M"] = data_unc[12].mean()

    if model == "Model2":
        data_unc_all = data_unc_all[
            ((data_unc_all[9] >= z_min) & (data_unc_all[9] <= z_max))
            | (data_unc_all[9] == 260)
        ]
        data_unc_all["alpha_bin"] = data_unc_all[alpha_col].round(1)
        grouped = data_unc_all.groupby("alpha_bin", as_index=False, dropna=False).mean()
        grouped = grouped.drop(columns=["alpha_bin"])
        data = grouped
        idx = data[data[9] == 260].index
        params["skip_idx"] = idx.tolist()
        if not idx.empty:
            use_idx = idx[0]
            if use_idx == 0 and len(idx) > 1:
                use_idx = idx[1]
                params["amax"] = data.iloc[use_idx, alpha_col]
            elif use_idx == 0:
                params["amax"] = 50
            else:
                params["amax"] = data.iloc[use_idx, alpha_col]
        else:
            params["amax"] = 50
    elif model == "V3":
        idx = data_unc[data_unc[9].isin([40, 120])].index
        params["skip_idx"] = idx.tolist()
        if not idx.empty:
            use_idx = idx[0]
            if use_idx == 0 and len(idx) > 1:
                use_idx = idx[1]
                params["amax"] = data_unc.iloc[use_idx, alpha_col]
            elif use_idx == 0:
                params["amax"] = 50
            else:
                params["amax"] = data_unc.iloc[use_idx, alpha_col]
        else:
            params["amax"] = 50
        data = data_unc
    else:
        raise ValueError(f"Unsupported model '{model}'.")

    keep_cols = [1, 5, 6, 7] + CPwu_column + CPwl_column
    keep_cols = sorted(set(keep_cols))
    return data.iloc[:, keep_cols]


def load_and_preprocess_data_corr(
    filename: Path,
    model: str,
    CPwu_column: List[int],
    CPwl_column: List[int],
    z_min: int = 340,
    z_max: int = 380,
    alpha_col: int = 1,
    tol: float = 0.05,
) -> pd.DataFrame:
    data = pd.read_csv(
        filename,
        sep="\t",
        header=None,
        skiprows=2,
        usecols=range(18),
    )
    data = data.apply(pd.to_numeric, errors="coerce")
    if model == "Model2":
        data = data[(data[13] >= z_min) & (data[13] <= z_max) | (data[13] == 260)]
        data["alpha_bin"] = data[alpha_col].round(1)
        grouped = data.groupby("alpha_bin", as_index=False, dropna=False).mean()
        grouped = grouped.drop(columns=["alpha_bin"])
        data = grouped
    keep_cols = [1, 2, 3, 4]
    keep_cols = sorted(set(keep_cols))
    return data.iloc[:, keep_cols]


def get_cm_from_alpha(alpha_query: np.ndarray, model: str, Re: str) -> np.ndarray:
    if model == "V3" and Re == "1e6":
        alpha_data = np.array([-10, -5, -2, 0, 2, 6, 10, 12, 14, 16, 18, 20, 22])
        cm_data = np.array(
            [
                5.90e-02,
                5.43e-02,
                2.99e-02,
                -1.34e-02,
                -5.34e-02,
                -8.27e-02,
                -7.64e-02,
                -8.01e-02,
                -1.19e-01,
                -1.48e-01,
                -1.61e-01,
                -1.69e-01,
                -1.75e-01,
            ]
        )
    elif model == "V3" and Re == "5e5":
        alpha_data = np.array([-2, 6, 10, 12, 14, 16, 18, 20, 22])
        cm_data = np.array(
            [
                1.86e-02,
                -8.64e-02,
                -7.86e-02,
                -8.82e-02,
                -1.89e-01,
                -1.49e-01,
                -1.55e-01,
                -1.52e-01,
                -1.57e-01,
            ]
        )
    elif model == "Model2" and Re == "1e6":
        alpha_data = np.array([-10, -5, -2, 0, 2, 6, 10, 12, 14, 16, 18, 20, 22])
        cm_data = np.array(
            [
                4.89e-02,
                3.20e-02,
                1.30e-02,
                -4.21e-02,
                -8.64e-02,
                -1.10e-01,
                -1.07e-01,
                -1.06e-01,
                -1.17e-01,
                -1.21e-01,
                -1.35e-01,
                -1.49e-01,
                -1.63e-01,
            ]
        )
    elif model == "Model2" and Re == "5e5":
        alpha_data = np.array([6, 10, 12, 14, 16, 18, 20, 22])
        cm_data = np.array(
            [
                -1.03e-01,
                -9.24e-02,
                -9.58e-02,
                -1.23e-01,
                -1.69e-01,
                -1.74e-01,
                -1.81e-01,
                -1.86e-01,
            ]
        )
    else:
        raise ValueError(f"Unsupported combination model={model}, Re={Re}.")

    spline = InterpolatedUnivariateSpline(alpha_data, cm_data, k=3, ext="extrapolate")
    alpha_clipped = np.clip(alpha_query, -10, 25)
    return spline(alpha_clipped)


def process_case(
    casename: str,
    alpha_min: float,
    alpha_max: float,
    method: str,
    transition: bool = True,
) -> pd.DataFrame:
    model, Re = parse_case_info(casename)
    params = get_model_parameters(model)

    cfd_results = data_cfd(model, Re, transition)

    case_dir = PROJECT_DIR / model / casename
    filename_unc_all = case_dir / f"unc_all_{casename}.txt"
    filename_unc = case_dir / f"unc_{casename}.txt"
    filename_corr = case_dir / f"corr_all_{casename}.txt"

    if not filename_unc_all.exists():
        raise FileNotFoundError(f"Missing file: {filename_unc_all}")
    if not filename_unc.exists():
        raise FileNotFoundError(f"Missing file: {filename_unc}")
    if not filename_corr.exists():
        raise FileNotFoundError(f"Missing file: {filename_corr}")

    data_unc = load_and_preprocess_data_unc(
        filename_unc_all,
        filename_unc,
        model,
        STRIPS["CPwu_column"],
        STRIPS["CPwl_column"],
        params,
    )
    data_corr = load_and_preprocess_data_corr(
        filename_corr,
        model,
        STRIPS["CPwu_column"],
        STRIPS["CPwl_column"],
    )

    params["factor"] = correction_factor_single_case(
        casename,
        model,
        STRIPS,
        alpha_min,
        alpha_max,
        method,
        provide_plot=False,
    )

    params["c_h"] = params["c1"] / params["h"]
    params["tau"] = 0.25 * params["c_h"]
    params["sigma"] = np.pi**2 / 48 * params["c_h"] ** 2
    params["eps"] = params["sigma"] * params["lambda"]
    params["beta"] = np.sqrt(1 - params["M"] ** 2)
    params["eps_alpha"] = 1.1 * params["beta"] / params["t_c"]

    params["factor_sc"] = params["sigma"] / params["beta"] ** 2
    params["factor_sb"] = params["lambda"] * params["sigma"] / params["beta"] ** 3
    params["factor_wb"] = (
        params["tau"] * (1 + 0.4 * params["M"] ** 2) / params["beta"] ** 2
    )

    data_unc.iloc[:, 3] = 0
    data_corr.iloc[:, 3] = 0

    postdata_presscorr = data_unc.copy().values
    cl_col = 2
    press_cols = np.arange(4, postdata_presscorr.shape[1])
    zerolift_press = np.array(
        [
            interp1d(
                postdata_presscorr[:, cl_col],
                postdata_presscorr[:, col],
                kind="linear",
                fill_value="extrapolate",
            )(0)
            for col in press_cols
        ]
    )
    postdata_presscorr[:, 4:] = postdata_presscorr[:, 4:] - zerolift_press

    n_x = len(STRIPS["x"])
    cpwu = postdata_presscorr[:, 4 : 4 + n_x]
    cpwl = postdata_presscorr[:, 4 + n_x : 4 + 2 * n_x]
    diff_cp = cpwl - cpwu
    postdata_presscorr[:, 2] = np.trapezoid(diff_cp, STRIPS["x"], axis=1) * (
        1 / params["c"]
    )

    postdata_stripcorr = postdata_presscorr[:, :4].copy()
    postdata_stripcorr = np.hstack(
        [postdata_stripcorr, np.zeros((postdata_stripcorr.shape[0], 6))]
    )
    postdata_stripcorr[:, 2] = postdata_stripcorr[:, 2] * params["factor"]
    postdata_c1_cdcorr = postdata_stripcorr.copy()

    cd_interp = InterpolatedUnivariateSpline(
        cfd_results["Alpha"],
        cfd_results["Cd"],
        k=3,
        ext="extrapolate",
    )
    extrap_idx = np.where(postdata_c1_cdcorr[:, 0] > params["amax"])[0]
    if extrap_idx.size > 0:
        cd_extrapolated = cd_interp(postdata_c1_cdcorr[extrap_idx, 0])
        postdata_c1_cdcorr[extrap_idx, 1] = cd_extrapolated

    postdata_c2_cmcorr = postdata_c1_cdcorr.copy()
    postdata_c2_cmcorr[:, 3] = get_cm_from_alpha(
        postdata_c2_cmcorr[:, 0],
        model,
        Re,
    )

    postdata_final = postdata_c2_cmcorr.copy()
    placeholder_low = 0
    placeholder_up = cd_interp(postdata_final[:, 0]) * 2
    placeholder_mid = cd_interp(postdata_final[:, 0])

    postdata_final[:, 0] = postdata_final[:, 0] + 57.3 * params["sigma"] / (
        params["beta"] * 2 * np.pi
    ) * (postdata_final[:, 2] + postdata_final[:, 3])

    params["eps_solid_blockage"] = (
        (2 - params["M"] ** 2)
        * params["factor_sb"]
        * (1 + params["eps_alpha"] * (postdata_final[:, 0] * np.pi / 180) ** 2)
    )
    params["eps_wake_blockage_measured"] = (2 - params["M"] ** 2) * (
        params["factor_wb"] * postdata_final[:, 1]
    )
    params["eps_wake_blockage_low"] = (2 - params["M"] ** 2) * (
        params["factor_wb"] * placeholder_low
    )
    params["eps_wake_blockage_mid"] = (2 - params["M"] ** 2) * (
        params["factor_wb"] * placeholder_mid
    )
    params["eps_wake_blockage_up"] = (2 - params["M"] ** 2) * (
        params["factor_wb"] * placeholder_up
    )

    postdata_final[:, 2] = postdata_final[:, 2] * (
        1
        - params["factor_sc"]
        + 5.25 * params["factor_sc"] ** 2
        - params["eps_solid_blockage"]
        - params["eps_wake_blockage_measured"]
    )
    postdata_final[:, 4] = postdata_final[:, 2] * (
        1
        - params["factor_sc"]
        + 5.25 * params["factor_sc"] ** 2
        - params["eps_solid_blockage"]
        - params["eps_wake_blockage_low"]
    )
    postdata_final[:, 5] = postdata_final[:, 2] * (
        1
        - params["factor_sc"]
        + 5.25 * params["factor_sc"] ** 2
        - params["eps_solid_blockage"]
        - params["eps_wake_blockage_mid"]
    )
    postdata_final[:, 6] = postdata_final[:, 2] * (
        1
        - params["factor_sc"]
        + 5.25 * params["factor_sc"] ** 2
        - params["eps_solid_blockage"]
        - params["eps_wake_blockage_up"]
    )

    postdata_final[:, 1] = postdata_final[:, 1] * (
        1 - params["eps_solid_blockage"] - params["eps_wake_blockage_measured"]
    )
    postdata_final[:, 7] = postdata_final[:, 1] * (
        1 - params["eps_solid_blockage"] - params["eps_wake_blockage_low"]
    )
    postdata_final[:, 8] = postdata_final[:, 1] * (
        1 - params["eps_solid_blockage"] - params["eps_wake_blockage_mid"]
    )
    postdata_final[:, 9] = postdata_final[:, 1] * (
        1 - params["eps_solid_blockage"] - params["eps_wake_blockage_up"]
    )

    postdata_final_df = pd.DataFrame(
        postdata_final,
        columns=[
            "alpha",
            "Cd",
            "Cl",
            "Cm",
            "Cl_low",
            "Cl_mid",
            "Cl_up",
            "Cd_low",
            "Cd_mid",
            "Cd_up",
        ],
    )
    postdata_final_df["alpha_unc"] = postdata_presscorr[:, 0].round(1)
    postdata_final_df["alpha_unc"] = postdata_final_df["alpha_unc"].round(1)

    grouped = postdata_final_df.groupby("alpha_unc")
    mean_df = grouped.mean()
    std_df = grouped.std().fillna(0)

    postdata_final_df = mean_df.reset_index()
    std_df = std_df.reset_index().rename(columns={"Cd": "std_Cd", "Cl": "std_Cl"})
    merged_df = postdata_final_df.merge(
        std_df[["alpha_unc", "std_Cd", "std_Cl"]], on="alpha_unc", how="left"
    )

    amax_idx = int(
        np.argmax(merged_df["alpha"].values >= params["amax"])
        if len(merged_df) > 0
        else 0
    )
    if len(merged_df) == 0:
        raise ValueError(f"No data remaining after processing case '{casename}'.")
    if merged_df.loc[min(amax_idx, len(merged_df) - 1), "alpha"] < params["amax"]:
        amax_idx = len(merged_df)

    skip_idx_list = params.get("skip_idx", [])
    start_idx = 1 if skip_idx_list and skip_idx_list[0] == 0 else 0
    last_measured_idx = min(amax_idx, len(merged_df) - 1)

    region = np.full(len(merged_df), "extrapolated", dtype=object)
    if start_idx > 0:
        region[:start_idx] = "skip"
    if last_measured_idx >= start_idx:
        region[start_idx : last_measured_idx + 1] = "measured"

    merged_df["region"] = region
    merged_df["is_measured"] = merged_df.index <= last_measured_idx
    merged_df["include_measured_plot"] = (merged_df.index >= start_idx) & (
        merged_df.index <= last_measured_idx
    )
    merged_df["casename"] = casename
    merged_df["model"] = model
    merged_df["Re"] = Re
    merged_df["transition"] = transition
    merged_df["correction_factor"] = params["factor"]
    merged_df["alpha_max_threshold"] = params["amax"]
    merged_df["alpha_min_input"] = alpha_min
    merged_df["alpha_max_input"] = alpha_max
    merged_df["method"] = method
    merged_df["skip_idx"] = ",".join(str(x) for x in skip_idx_list)
    merged_df["start_idx"] = start_idx
    merged_df["last_measured_idx"] = last_measured_idx

    return merged_df


def save_processed_data(df: pd.DataFrame, casename: str) -> Path:
    ensure_directory(PROCESSED_DATA_DIR)
    output_path = PROCESSED_DATA_DIR / f"{casename}_processed.csv"
    df.to_csv(output_path, index=False)
    return output_path


def process_cases(
    cases: Iterable[str],
    alpha_min: float,
    alpha_max: float,
    method: str,
) -> List[Tuple[str, Path]]:
    outputs = []
    for casename in cases:
        processed_df = process_case(casename, alpha_min, alpha_max, method)
        output_path = save_processed_data(processed_df, casename)
        outputs.append((casename, output_path))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process CFD vs experimental datasets into CSV files.",
    )
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        help="Case identifier to process (can be provided multiple times).",
    )
    parser.add_argument(
        "--alpha-min",
        type=float,
        default=DEFAULT_ALPHA_MIN,
        help="Minimum alpha for correction factor computation.",
    )
    parser.add_argument(
        "--alpha-max",
        type=float,
        default=DEFAULT_ALPHA_MAX,
        help="Maximum alpha for correction factor computation.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=DEFAULT_METHOD,
        help="Optimization method for correction factor.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = args.cases if args.cases else DEFAULT_CASES
    process_cases(cases, args.alpha_min, args.alpha_max, args.method)


if __name__ == "__main__":
    main()
