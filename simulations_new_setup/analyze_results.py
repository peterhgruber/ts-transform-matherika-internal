# ------------------------------------------------------------------------------
# Innosuisse Project: Usability of Transformer Models for Modelling Commodity Markets
# Date: July 16, 2025
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
from pathlib import Path
import yaml


# ------------------------------------------------------------------------------
# CLI input
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", required=True)
parser.add_argument("--target_type", choices=["prices", "returns"], default="prices")
parser.add_argument("--job_file", default="jobs/job_00.yaml")
parser.add_argument("--sim_name", default="sim_00")

args = parser.parse_args()

job_file_path = Path(args.job_file)
job_name = job_file_path.stem

model_name = args.model_name
target_type = args.target_type
model_base = model_name.split("_model")[0]
model_variant = model_name.replace(f"{model_base}_model_", "")


# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
forecast_dir = f"forecasts/{model_base}/{model_variant}/{job_name}_{args.sim_name}/{target_type}"
results_dir = f"results/{model_base}_results/{model_variant}/{job_name}_{args.sim_name}/{target_type}"

tables_dir = os.path.join(results_dir, "tables")
plots_dir_forecast = os.path.join(results_dir, "plots", "forecasts")
plots_dir_kde = os.path.join(results_dir, "plots", "kdes")
plots_dir_cdf = os.path.join(results_dir, "plots", "cdfs")
plots_dir_kl = os.path.join(results_dir, "plots", "kl_divergence")

for folder in [tables_dir, plots_dir_forecast, plots_dir_kde, plots_dir_cdf, plots_dir_kl]:
    os.makedirs(folder, exist_ok=True)

series_dir = f"data/{args.sim_name}/{target_type}_simulated_series"
paths_dir = f"data/{args.sim_name}/{target_type}_simulated_paths"


# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------
with open(args.job_file, "r") as f:
    config = yaml.safe_load(f)

context_lengths = config["context_lengths"]

selected_days = config.get("selected_days", [0, 10, 20]) # fallback default

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
    if target_type == "prices":
        daily_returns = samples[:, 1:] / samples[:, :-1] - 1
    else:
        daily_returns = samples

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

summary_df = pd.DataFrame(summary_rows).round(2)
pivot_forecast = summary_df.pivot(
    index=["context_length", "dgp_type"],
    columns="day",
    values=["mean_return (%)", "std_return (%)"]
)
pivot_forecast = format_pivot_table(pivot_forecast, selected_days, dgp_order=dgp_types_all)
dataframe_to_latex(pivot_forecast, os.path.join(tables_dir, "forecast_table.tex"))

percentile_df = pd.DataFrame(percentile_rows)
pivot_percentiles = percentile_df.pivot_table(
    index=["context_length", "dgp_type"],
    columns=["day", "percentile"],
    values="return (%)"
).round(2)

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

        _, _, _, model_samples, _ = all_results[key]
        model_returns = model_samples if target_type == "returns" else model_samples[:, 1:] / model_samples[:, :-1] - 1

        dgp_path = os.path.join(paths_dir, f"{target_type}_{dgp}_seed42.npy")
        if not os.path.exists(dgp_path):
            print(f"[SKIP] Missing DGP: {dgp_path}")
            continue

        dgp_samples = np.load(dgp_path)
        dgp_returns = dgp_samples if target_type == "returns" else dgp_samples[:, 1:] / dgp_samples[:, :-1] - 1

        for day_index in selected_days:
            try:
                p = dgp_returns[:, day_index]
                q = model_returns[:, day_index]
            except IndexError:
                print(f"[SKIP] IndexError: {dgp}, context {context}, day {day_index}")
                continue

            if not np.isfinite(p).all() or not np.isfinite(q).all():
                print(f"[SKIP] Non-finite values for {dgp}, context {context}, day {day_index}")
                continue

            kl = compute_kl_divergence(p, q)
            kl_results.append({
                "context_length": context,
                "dgp_type": dgp,
                "day": f"Day {day_index + 2}",
                "kl_divergence": kl
            })

if len(kl_results) == 0:
    print("[ERROR] KL results are empty. Check forecasts or DGP files.")
    exit(1)

kl_df = pd.DataFrame(kl_results).round(4)
pivot_kl = kl_df.pivot(index=["context_length", "dgp_type"], columns="day", values="kl_divergence")
pivot_kl = format_pivot_table(pivot_kl, selected_days, dgp_order=dgp_types_kl)
dataframe_to_latex(pivot_kl, os.path.join(tables_dir, "kl_divergence_table.tex"))


# ------------------------------------------------------------------------------
# Plotting: Forecasts, KDEs, CDFs
# ------------------------------------------------------------------------------
for (context, dgp), (low, median, high, samples, base_price) in all_results.items():
    series_path = os.path.join(series_dir, f"{target_type}_{dgp}_seed42.csv")
    if not os.path.exists(series_path):
        continue

    series = pd.read_csv(series_path).squeeze()
    series.index = range(len(series))

    forecast_path = os.path.join(plots_dir_forecast, f"{dgp}_ctx{context}_forecast.png")
    kdes_path = os.path.join(plots_dir_kde, f"{dgp}_ctx{context}_kdes.png")
    cdfs_path = os.path.join(plots_dir_cdf, f"{dgp}_ctx{context}_cdfs.png")
    
    is_price_data = (target_type == "prices")

    plot_forecast(
    series=series,
    low=low,
    median=median,
    high=high,
    dgp_type=dgp,
    context_length=context,
    path=forecast_path,
    is_price_data=is_price_data
    )

    plot_daily_return_kdes(
        samples=samples,
        selected_days=selected_days,
        dgp_type=dgp,
        context_length=context,
        path=kdes_path,
        is_price_data=is_price_data
    )

    plot_daily_return_cdfs(
        samples=samples,
        selected_days=selected_days,
        dgp_type=dgp,
        context_length=context,
        path=cdfs_path,
        is_price_data=is_price_data
    )


# ------------------------------------------------------------------------------
# Return density comparison plots (KL)
# ------------------------------------------------------------------------------
for (context, dgp), (_, _, _, model_samples, _) in all_results.items():
    if dgp not in dgp_types_kl:
        continue

    dgp_path = os.path.join(paths_dir, f"{target_type}_{dgp}_seed42.npy")
    if not os.path.exists(dgp_path):
        continue

    dgp_samples = np.load(dgp_path)
    model_returns = model_samples if target_type == "returns" else model_samples[:, 1:] / model_samples[:, :-1] - 1
    dgp_returns = dgp_samples if target_type == "returns" else dgp_samples[:, 1:] / dgp_samples[:, :-1] - 1

    output_path = os.path.join(plots_dir_kl, f"{dgp}_ctx{context}_kl_plot.png")

    plot_multiple_return_kde_comparison(
    dgp_samples=dgp_returns,
    model_samples=model_returns,
    selected_days=selected_days,
    dgp_type=dgp,
    context_length=context,
    path=output_path,
    is_price_data=False  # Already pre-computed as returns
)


