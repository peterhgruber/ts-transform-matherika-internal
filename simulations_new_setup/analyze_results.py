# ------------------------------------------------------------------------------
# Innosuisse Project: Usability of Transformer Models for Modelling Commodity Markets
# Date: July 9, 2025
# Authors: Peter Gruber, Alessandro Dodon
#
# Script: Analyze forecast results (tables, plots, KL divergence)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Packages
# ------------------------------------------------------------------------------
import argparse
import os
import pickle
import numpy as np
import pandas as pd
from utils.evaluation import compute_kl_divergence, format_pivot_table, dataframe_to_latex
from utils.plotting import *


# ------------------------------------------------------------------------------
# CLI input for model name
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", required=True)
args = parser.parse_args()

model_name = args.model_name
model_base = model_name.split("_model")[0]
model_variant = model_name.replace(f"{model_base}_model_", "")


# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
forecast_dir = f"forecasts/{model_base}/{model_variant}"
results_dir = f"results/{model_base}_results/{model_variant}"
tables_dir = os.path.join(results_dir, "tables")
plots_dir = os.path.join(results_dir, "plots")
paths_dir = "data/simulated_paths"
prices_dir = "data/simulated_prices"

os.makedirs(tables_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)


# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------
context_lengths = [22, 66, 252]
selected_days = [0, 10, 20]
ordered_days = [f"Day {d+2}" for d in selected_days]
ordered_percentiles = ["p1%", "p5%", "p25%", "p50%", "p75%", "p95%", "p99%"]
percentile_values = [int(p.replace("p", "").replace("%", "")) for p in ordered_percentiles]

dgp_types_all = ["gbm_low_vol", "gbm_high_vol", "t_garch", "mixture_normal", "seasonal", "constant", "linear"]
dgp_types_kl = ["gbm_low_vol", "gbm_high_vol", "t_garch", "mixture_normal", "seasonal"]


# ------------------------------------------------------------------------------
# Load forecast results
# ------------------------------------------------------------------------------
all_results = {}

for context in context_lengths:
    for dgp in dgp_types_all:
        file_path = os.path.join(forecast_dir, f"{dgp}_ctx{context}_seed42.pkl")
        if not os.path.exists(file_path):
            continue
        with open(file_path, "rb") as f:
            result = pickle.load(f)
        all_results[(context, dgp)] = result


# ------------------------------------------------------------------------------
# Compute forecast summary tables
# ------------------------------------------------------------------------------
summary_rows = []
percentile_rows = []

for (context, dgp), (low, median, high, samples, base_price) in all_results.items():
    daily_returns = samples[:, 1:] / samples[:, :-1] - 1

    for day in selected_days:
        returns = daily_returns[:, day]

        summary_rows.append({
            "context_length": context,
            "dgp_type": dgp,
            "day": f"Day {day + 2}",
            "mean_return (%)": np.mean(returns) * 100,
            "std_return (%)": np.std(returns) * 100
        })

        for p_label, p_val in zip(ordered_percentiles, np.percentile(returns, percentile_values)):
            percentile_rows.append({
                "context_length": context,
                "dgp_type": dgp,
                "day": f"Day {day + 2}",
                "percentile": p_label,
                "return (%)": p_val * 100
            })

# Save summary and percentiles
summary_df = pd.DataFrame(summary_rows).round(2)
pivot_forecast = summary_df.pivot(
    index=["context_length", "dgp_type"],
    columns="day",
    values=["mean_return (%)", "std_return (%)"]
)
pivot_forecast = format_pivot_table(pivot_forecast, selected_days, dgp_order=dgp_types_all)
dataframe_to_latex(pivot_forecast, os.path.join(tables_dir, "forecast_table.tex"))

# Create and save percentiles table
percentile_df = pd.DataFrame(percentile_rows)

pivot_percentiles = percentile_df.pivot_table(
    index=["context_length", "dgp_type"],
    columns=["day", "percentile"],
    values="return (%)"
).round(2)

# Ensure consistent column ordering
ordered_columns = pd.MultiIndex.from_product([ordered_days, ordered_percentiles])
pivot_percentiles = pivot_percentiles.reindex(columns=ordered_columns)

pivot_percentiles = format_pivot_table(pivot_percentiles, selected_days, dgp_order=dgp_types_all)
dataframe_to_latex(pivot_percentiles, os.path.join(tables_dir, "percentiles_table.tex"))


# ------------------------------------------------------------------------------
# KL divergence
# ------------------------------------------------------------------------------
kl_results = []

for context in context_lengths:
    for dgp in dgp_types_kl:
        key = (context, dgp)
        if key not in all_results:
            continue

        _, _, _, chronos_samples, _ = all_results[key]
        chronos_returns = chronos_samples[:, 1:] / chronos_samples[:, :-1] - 1

        dgp_file = os.path.join(paths_dir, f"{dgp}_seed42.npy")
        if not os.path.exists(dgp_file):
            print(f"[SKIP] Missing DGP: {dgp_file}")
            continue

        dgp_samples = np.load(dgp_file)
        dgp_returns = dgp_samples[:, 1:] / dgp_samples[:, :-1] - 1

        for day_index in selected_days:
            p = dgp_returns[:, day_index]
            q = chronos_returns[:, day_index]

            if not np.isfinite(p).all() or not np.isfinite(q).all():
                print(f"[WARNING] KL divergence invalid values for {dgp}, context {context}, day {day_index + 2}")
                continue

            kl = compute_kl_divergence(p, q)
            kl_results.append({
                "context_length": context,
                "dgp_type": dgp,
                "day": f"Day {day_index + 2}",
                "kl_divergence": kl
            })

kl_df = pd.DataFrame(kl_results).round(4)
pivot_kl = kl_df.pivot(index=["context_length", "dgp_type"], columns="day", values="kl_divergence")
pivot_kl = format_pivot_table(pivot_kl, selected_days, dgp_order=dgp_types_kl)
dataframe_to_latex(pivot_kl, os.path.join(tables_dir, "kl_divergence_table.tex"))


# ------------------------------------------------------------------------------
# Plot forecasts and KDE/CDFs
# ------------------------------------------------------------------------------
for (context, dgp), (low, median, high, samples, base_price) in all_results.items():
    price_path = os.path.join(prices_dir, f"{dgp}_seed42.csv")
    if not os.path.exists(price_path):
        continue

    series = pd.read_csv(price_path).squeeze()
    series.index = range(len(series))
    base_path = os.path.join(plots_dir, f"{dgp}_ctx{context}")

    plot_forecast(
    series=series,
    low=low,
    median=median,
    high=high,
    dgp_type=dgp,
    context_length=context,
    path=base_path + "_forecast.png"
    )

    plot_daily_return_kdes(
        samples=samples,
        selected_days=selected_days,
        dgp_type=dgp,
        context_length=context,
        path=base_path + "_kdes.png"
    )

    plot_daily_return_cdfs(
        samples=samples,
        selected_days=selected_days,
        dgp_type=dgp,
        context_length=context,
        path=base_path + "_cdfs.png"
    )


# ------------------------------------------------------------------------------
# Return density comparison plots
# ------------------------------------------------------------------------------
for (context, dgp), (_, _, _, model_samples, _) in all_results.items():
    if dgp not in dgp_types_kl:
        continue

    dgp_path = os.path.join(paths_dir, f"{dgp}_seed42.npy")
    if not os.path.exists(dgp_path):
        continue

    dgp_samples = np.load(dgp_path)
    output_path = os.path.join(plots_dir, f"{dgp}_ctx{context}_kl_plot.png")

    plot_multiple_return_kde_comparison(
        dgp_samples=dgp_samples,
        model_samples=model_samples,
        selected_days=selected_days,
        dgp_type=dgp,
        context_length=context,
        path=output_path
    )
