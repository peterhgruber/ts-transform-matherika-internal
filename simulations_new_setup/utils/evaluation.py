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
def summarize_selected_day_returns(samples, selected_days, is_price_data=True):
    """
    Compute mean and std of daily returns for selected forecast days.

    Parameters:
    - samples: np.ndarray, shape (n_samples, n_days)
    - selected_days: list of int
    - is_price_data: bool, whether samples are prices or already returns
    """
    if is_price_data:
        daily_returns = samples[:, 1:] / samples[:, :-1] - 1
    else:
        daily_returns = samples

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
# Latex conversion for DataFrame (Flexible Index Support)
# ------------------------------------------------------------------------------
def dataframe_to_latex(dataframe, output_path, preserve_index_order=False):
    """
    Converts a DataFrame with multi-index to a LaTeX tabular format.

    Parameters:
        dataframe (pd.DataFrame): Pivoted and optionally pre-sorted DataFrame.
        output_path (Path): Output .tex file location.
        preserve_index_order (bool): If True, row order is preserved as-is.
                                     If False, standard alphabetical sort is applied.
    """
    def latex_escape(text):
        if isinstance(text, str):
            return text.replace('_', '\\_').replace('%', '\\%')
        return text

    table_copy = dataframe.copy().round(2)

    # Escape column names
    if isinstance(table_copy.columns, pd.MultiIndex):
        table_copy.columns = pd.MultiIndex.from_tuples([
            tuple(latex_escape(col) for col in col_tuple)
            for col_tuple in table_copy.columns
        ])
        column_tuples = table_copy.columns.tolist()
        day_groups = [day for day, _ in column_tuples]
        subheaders = [label for _, label in column_tuples]
    else:
        table_copy.columns = [latex_escape(col) for col in table_copy.columns]
        day_groups = table_copy.columns
        subheaders = ["Value"] * len(day_groups)

    # Escape row index values
    table_copy.index = pd.MultiIndex.from_tuples([
        tuple(latex_escape(val) for val in row)
        for row in table_copy.index
    ], names=table_copy.index.names)

    # Control row sorting
    if not preserve_index_order:
        sort_levels = [level for level in ["model_name", "dgp_type", "context_length"] if level in table_copy.index.names]
        table_copy = table_copy.sort_index(level=sort_levels)

    # Convert values to strings
    table_copy = table_copy.applymap(lambda x: f"{x:.2f}")

    # Column count for header
    col_count = table_copy.shape[1]
    num_index_cols = len(table_copy.index.names)
    header = "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{" + "l" * num_index_cols + "r" * col_count + "}\n\\toprule\n"

    # Column group headers (top row)
    header += " & " * num_index_cols
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

    # Subheaders (second row)
    index_headers = [col.replace("_", "\\_") for col in table_copy.index.names]
    header += " & ".join(index_headers) + " & " + " & ".join(subheaders) + " \\\\\n\\midrule\n"

    # Row data
    lines = []
    last_key = None

    for index_tuple, row in table_copy.iterrows():
        index_names = table_copy.index.names
        index_dict = dict(zip(index_names, index_tuple))
        current_key = tuple(index_dict.get(k, "") for k in ["model_name", "dgp_type"])

        if last_key and current_key != last_key:
            lines.append("\\midrule")
        last_key = current_key

        index_values = [str(index_dict.get(col, "")) for col in index_names]
        row_values = " & ".join(row.tolist())
        lines.append(f"{' & '.join(index_values)} & {row_values} \\\\")

    footer = "\n\\bottomrule\n\\end{tabular}\n}"

    with open(output_path, "w") as f:
        f.write(header + "\n".join(lines) + footer)


