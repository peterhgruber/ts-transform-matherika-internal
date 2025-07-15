# ------------------------------------------------------------------------------
# Innosuisse Project: Usability of Transformer Models for Modelling Commodity Markets
# Date: June 29, 2025
# Authors: Peter Gruber (peter.gruber@usi.ch), Alessandro Dodon (alessandro.dodon@usi.ch)
#
# This script defines the simulation functions used in all experiments.
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Packages
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')


# PART 1: COMPUTING METRICS


# ------------------------------------------------------------------------------
# Additional metrics: mean, std for selected days
# ------------------------------------------------------------------------------
def summarize_selected_day_returns(samples, selected_days):
    """
    Print mean and standard deviation of forecasted daily returns
    for specific forecast days.

    Parameters:
    - samples: np.ndarray of shape (n_samples, n_days)
        Matrix of simulated forecast paths. Each row represents one full path
        of forecasted prices over `n_days`, and each column represents a specific
        forecast day (e.g. Day 1, Day 2, ..., Day 22).
    
    - selected_days: list of int
        Indices referring to the day of the return to analyze, i.e. index 0 means
        return from Day 1 to Day 2, index 10 means return from Day 11 to Day 12.

    Returns:
    - summary: dict
        Dictionary mapping each selected day index to its mean and std of returns.

    Notes:
    - Daily returns are computed as (price_t+1 / price_t - 1) for each row (path).
    - This yields a return matrix of shape (n_samples, n_days - 1).
    - For each `day` in `selected_days`, we extract the corresponding column of
      the return matrix and compute statistics across all paths (i.e., row-wise).
    - The mean and standard deviation are printed in percentage format.
    """
    daily_returns = samples[:, 1:] / samples[:, :-1] - 1
    summary = {}

    print("=== Return Summary for Selected Days ===")
    for day in selected_days:
        returns = daily_returns[:, day]
        mean_r = np.mean(returns)
        std_r = np.std(returns)

        print(f"\nReturn Day {day + 2}:")
        print(f"  Mean:  {mean_r:.4%}")
        print(f"  Std:   {std_r:.4%}")

        summary[day] = {
            "mean": mean_r,
            "std": std_r
        }

    return summary


# ------------------------------------------------------------------------------
# KL Divergence Test 
# ------------------------------------------------------------------------------
def compute_kl_divergence(p_samples, q_samples, num_bins=200):
    """
    Compute the KL divergence D_KL(P || Q) using KDE over a common support.

    Parameters:
    - p_samples: np.ndarray, samples from true DGP (reference)
    - q_samples: np.ndarray, samples from Model forecast (approximation)
    - num_bins: int, number of evaluation points for density comparison

    Returns:
    - kl_value: float, KL divergence
    """
    # Define common evaluation range
    support_min = min(p_samples.min(), q_samples.min())
    support_max = max(p_samples.max(), q_samples.max())
    x_grid = np.linspace(support_min, support_max, num_bins)

    # Estimate KDEs
    kde_p = gaussian_kde(p_samples)
    kde_q = gaussian_kde(q_samples)

    p_density = kde_p(x_grid)
    q_density = kde_q(x_grid)

    # Avoid divide-by-zero
    epsilon = 1e-8
    p_density = np.clip(p_density, epsilon, None)
    q_density = np.clip(q_density, epsilon, None)

    # Compute KL divergence
    kl_value = entropy(p_density, q_density)

    return kl_value


# PART 2: EXPORTING RESULTS


# ------------------------------------------------------------------------------
# Pivot Table 
# ------------------------------------------------------------------------------
def format_pivot_table(raw_table, selected_days, dgp_order=None):
    """
    Reorders a pivot table both in columns (day ordering) and rows (dgp_type order).

    Parameters:
    - raw_table: pd.DataFrame, a pivoted table with multi-index rows
    - selected_days: list of int, e.g. [0, 10, 20]
    - dgp_order: list of str, custom order for dgp_type values (optional)

    Returns:
    - formatted_table: pd.DataFrame
    """
    # Reorder columns
    ordered_days = [f"Day {day + 2}" for day in sorted(selected_days)]

    if isinstance(raw_table.columns, pd.MultiIndex):
        raw_table = raw_table.reorder_levels([1, 0], axis=1).sort_index(
            axis=1,
            level=0,
            key=lambda x: pd.Categorical(x, categories=ordered_days, ordered=True)
        )
    else:
        raw_table = raw_table[ordered_days]

    # Reorder rows
    if dgp_order:
        raw_table = raw_table.sort_index(
            level=["context_length", "dgp_type"],
            key=lambda idx: idx.map(lambda x: dgp_order.index(x) if x in dgp_order else -1)
            if idx.name == "dgp_type" else idx
        )

    return raw_table


# ------------------------------------------------------------------------------
# Latex conversion for DataFrame
# ------------------------------------------------------------------------------
def dataframe_to_latex(dataframe, output_path):
    """
    Converts a pivoted DataFrame (with possibly multi or single index columns)
    into a clean academic-style LaTeX table.

    Required LaTeX packages (add to your preamble in Overleaf):
    \\usepackage{booktabs}     % for \\toprule, \\midrule, \\bottomrule
    \\usepackage{graphicx}     % for \\resizebox
    \\usepackage{array}        % for column formatting tweaks
    \\usepackage{caption}      % for nicer table captions (optional)
    """
    def latex_escape(text):
        if isinstance(text, str):
            return text.replace('_', '\\_').replace('%', '\\%')
        return text

    table_copy = dataframe.copy().round(2)

    # Escape and analyze columns
    if isinstance(table_copy.columns, pd.MultiIndex):
        table_copy.columns = pd.MultiIndex.from_tuples([
            tuple(latex_escape(col) for col in col_tuple)
            for col_tuple in table_copy.columns
        ])
        column_tuples = table_copy.columns.tolist()
        day_groups = []
        subheaders = []
        for day, label in column_tuples:
            day_groups.append(day)
            subheaders.append(label)
    else:
        # Single-level columns
        table_copy.columns = [latex_escape(col) for col in table_copy.columns]
        day_groups = table_copy.columns
        subheaders = ["KL divergence"] * len(day_groups)

    # Escape index values
    table_copy.index = pd.MultiIndex.from_tuples([
        (row[0], latex_escape(row[1])) for row in table_copy.index
    ], names=["context_length", "dgp_type"])

    # Convert values to strings with 2 decimals
    table_copy = table_copy.applymap(lambda x: f"{x:.2f}")

    # Column count
    col_count = table_copy.shape[1]

    # LaTeX preamble
    header = "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{" + "ll" + "r" * col_count + "}\n\\toprule\n"

    # Compose group header
    header += " & " * 2  # for index columns
    last_group = None
    span = 0
    spans = []

    for group in day_groups:
        if group == last_group:
            span += 1
        else:
            if last_group is not None:
                spans.append((last_group, span))
            last_group = group
            span = 1
    spans.append((last_group, span))

    for group, width in spans:
        header += f"\\multicolumn{{{width}}}{{l}}{{{group}}} & "
    header = header.rstrip("& ") + " \\\\\n"

    # Compose subheader
    header += "context\\_length & dgp\\_type & " + " & ".join(subheaders) + " \\\\\n\\midrule\n"

    # Compose rows
    lines = []
    last_context = None
    for (context_length, dgp_type), row in table_copy.iterrows():
        if last_context is not None and context_length != last_context:
            lines.append("\\midrule")
        last_context = context_length
        row_values = " & ".join(row.tolist())
        lines.append(f"{context_length} & {dgp_type} & {row_values} \\\\")

    footer = "\n\\bottomrule\n\\end{tabular}\n}"

    with open(output_path, "w") as f:
        f.write(header + "\n".join(lines) + footer)
