from typing import Any

import numpy as np
import pandas as pd
from bioemu_benchmarks.logger import get_logger
from sklearn.neighbors import KernelDensity

LOGGER = get_logger(__name__)

K_BOLTZMANN = 0.001987203599772605  # Boltzmann constant in kcal / mol / K


def _foldedness_from_fnc(
    fnc: np.ndarray, p_fold_thr: float, steepness: float
) -> np.ndarray:
    """
    Compute foldedness from fraction of native contacts (FNC).

    Args:
        fnc: Fraction of native contacts.
        p_fold_thr: FNC that has foldedness 0.5.
        steepness: Steepness of the sigmoid function.

    Returns:
        Foldedness values.
    """
    return 1 / (1 + np.exp(-2 * steepness * (fnc - p_fold_thr)))


def _compute_threshold(
    fnc: np.ndarray,
    min_abs_thr: int = 45,
    max_abs_thr: int = 90,
    margin_ratio_low: float = 0.2,
    margin_ratio_high: float = 0.3,
    bandwidth: float = 0.03,
) -> float:
    """
    Compute an adaptive threshold based on kernel density estimation (KDE) of the input data.

    This function applies KDE to the input function (fnc) and identifies a threshold value that can
    be used to distinguish between different states or conditions in the data. The threshold is
    computed within a specified range, which is adjusted based on the input data range and specified
    margin ratios. The function returns the computed threshold and the KDE score for the histogram
    bins.

    Args:
        fnc: An array containing the input data for which the threshold is to be computed.
        min_abs_thr: The minimum absolute threshold value as an integer percentage.
        max_abs_thr: The maximum absolute threshold value as an integer percentage.
        margin_ratio_low: Margin ratio used to adjust lower bound of threshold search range.
        margin_ratio_high: Margin ratio used to adjust upper bound of threshold search range.
        bandwidth: The bandwidth used for the KDE, as a ratio of the data range.

    Returns:
        Selected folded / unfolded threshold.
    """
    fnc_hist = np.histogram(fnc, bins=100, density=True, range=(0, 1))

    min_data = min(fnc)
    max_data = max(fnc)
    data_range = max_data - min_data
    # Compute Kernel Density
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth * data_range).fit(
        fnc.reshape(-1, 1)
    )
    kde_score: np.ndarray = kde.score_samples(fnc_hist[1].reshape(-1, 1))

    max_thr = int(min(max_data * 100 - (margin_ratio_high) * data_range * 100, max_abs_thr))
    min_thr = int(max(min_data * 100 + (margin_ratio_low) * data_range * 100, min_abs_thr))

    if min_thr >= max_thr:
        LOGGER.info("Warning: min_thr (%d) >= max_thr (%d)", min_thr, max_thr)
        if min_thr == min_abs_thr:
            return 1
        elif max_thr == max_abs_thr:
            return 0
        else:
            raise ValueError(f"min_thr ({min_thr}) >= max_thr ({max_thr})")

    relative_index = np.argmin(kde_score[min_thr:max_thr])
    threshold: float = (float(min_thr) + relative_index) / len(kde_score)

    return threshold


def _compute_dG(
    fnc: np.ndarray, temperature: float, p_fold_thr: float, steepness: float
) -> float:
    """Compute dG from sigmoid of fraction of native contacts"""
    p_fold_from_stat = _foldedness_from_fnc(
        fnc, p_fold_thr=p_fold_thr, steepness=steepness
    ).mean()
    p_fold_from_stat = np.clip(p_fold_from_stat, 1e-10, 1 - 1e-10)

    ratio = p_fold_from_stat / (1 - p_fold_from_stat)
    ratio = np.clip(ratio, 1e-10, 1e10)

    dG = -np.log(ratio) * K_BOLTZMANN * temperature  # default temperature 295 K
    return dG


def compute_dg_ddg_from_fnc(
    *,
    dict_fnc: dict[str, np.ndarray],
    system_info: pd.DataFrame,
    temperature: float = 295.0,
    p_fold_thr: float = 0.5,
    steepness: float = 10.0,
) -> pd.DataFrame:
    """
    Compute dG and ddG for a collection of systems based on their native contact scores.

    Args:
        dict_fnc: Dictionary with arrays containing fraction of native contact values for the
          different systems as entries and test case IDs as keys.
        system_info: Data frame containing benchmark information.
        temperature: Temperature used for free energy computation in Kelvin. Default: 295 K.
        p_fold_thr: Threshold for foldedness, i.e., FNC value that corresponds to foldedness 0.5,
            used as an offset in the sigmoid function. Default: 0.5.
        steepness: Steepness of the sigmoid function used to compute foldedness from FNC values.
            Default: 10.0.
    Returns:
        Data frame containing computed dG and ddGs, as well as experimental reference values and
        uncertainties.
    """
    free_energy_results: dict[str, dict[str, Any]] = {}

    # Compute corresponding dG from generated samples.
    for test_case in dict_fnc:
        # Load fraction of native contacts.
        fnc = dict_fnc[test_case]

        # Extract basic system info from benchmark definitions.
        sequence_dataframe = system_info[system_info.name == test_case]
        free_energy_results[test_case] = sequence_dataframe.to_dict(orient="records")[0]

        # Store temperature.
        free_energy_results[test_case]["temperature"] = temperature

        # Store number of samples and warn if below target value.
        num_samples_target = free_energy_results[test_case]["num_samples"]
        num_samples = len(fnc)
        if num_samples < 0.7 * num_samples_target:
            LOGGER.warning(
                "Number of samples for %s below recommendation (%s/%s).",
                test_case,
                num_samples,
                num_samples_target,
            )
        free_energy_results[test_case]["num_samples"] = num_samples

        # Compute dG and store.
        dg_pred = _compute_dG(
            fnc, temperature=temperature, p_fold_thr=p_fold_thr, steepness=steepness
        )
        free_energy_results[test_case]["dg_pred"] = dg_pred

    # Compute ddGs from dGs.
    for test_case in free_energy_results:
        test_case_wt = free_energy_results[test_case]["name_wt"]

        # Do not compute ddGs within the same system.
        if test_case == test_case_wt:
            continue

        if test_case_wt not in free_energy_results:
            LOGGER.warning(
                "Could not find wild type results for %s for ddG", test_case_wt
            )
            continue

        free_energy_results[test_case]["ddg_pred"] = (
            free_energy_results[test_case]["dg_pred"]
            - free_energy_results[test_case_wt]["dg_pred"]
        )

    return pd.DataFrame(free_energy_results).T
