# statistics/wilcoxon_curve_compare.py

import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from pathlib import Path


def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def collect_eer_by_n(data, n):
    """
    data: dict loaded from JSON
    n: int
    return: dict {user: eer}
    """
    eer_dict = {}
    for user, records in data.items():
        if str(n) in records:
            eer = records[str(n)].get("EER", None)
            if eer is not None:
                eer_dict[user] = float(eer)
    return eer_dict


def wilcoxon_curve_test(
    json_a_path,
    json_b_path,
    n_targets,
    label_a="Method A",
    label_b="Method B",
    alpha=0.1,
):
    """
    Replicates the exact logic of sig_diffs but for n-based score fusion curves.

    Parameters
    ----------
    json_a_path : str
        Path to first JSON (e.g. XYPlot)
    json_b_path : str
        Path to second JSON (e.g. CDF)
    n_targets : list[int]
        List of n values to test
    label_a, label_b : str
        Names shown in result table
    alpha : float
        Significance level
    """

    data_a = load_json(json_a_path)
    data_b = load_json(json_b_path)

    results = []

    for n in n_targets:
        eer_a = collect_eer_by_n(data_a, n)
        eer_b = collect_eer_by_n(data_b, n)

        # paired users only
        common_users = sorted(set(eer_a.keys()) & set(eer_b.keys()))

        if len(common_users) < 2:
            print(f"[Skip] n={n}: not enough paired users")
            continue

        vec_a = np.array([eer_a[u] for u in common_users])
        vec_b = np.array([eer_b[u] for u in common_users])


        mean_a = np.mean(vec_a)
        mean_b = np.mean(vec_b)
        abs_diff = abs(mean_b - mean_a)

        w_stat, p_value = wilcoxon(vec_a, vec_b)

        results.append({
            "Comparison": f"{label_a} vs {label_b}",
            "n": n,
            "Users": len(common_users),
            f"Mean EER ({label_a})": round(mean_a * 100, 2),
            f"Mean EER ({label_b})": round(mean_b * 100, 2),
            "Absolute Diff (%)": round(abs_diff * 100, 2),
            "Wilcoxon p-value": p_value,
            "Statistically Significant": p_value <= alpha,
        })

    df = pd.DataFrame(results)
    return df
