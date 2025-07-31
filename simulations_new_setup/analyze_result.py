# ------------------------------------------------------------------------------
# Packages
# ------------------------------------------------------------------------------
import pickle
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from utils.evaluation import compute_kl_divergence
from utils.plotting import (
    plot_forecast,
    plot_daily_return_kdes,
    plot_daily_return_cdfs,
    plot_multiple_return_kde_comparison
)


# ------------------------------------------------------------------------------
# Read .txt runfile
# ------------------------------------------------------------------------------
if len(sys.argv) < 2:
    raise ValueError("Usage: python analyze_result.py <runfile.txt>")

runfile_path = Path(sys.argv[1])
if not runfile_path.exists():
    raise FileNotFoundError(f"Runfile not found: {runfile_path}")

run_config = {}
with open(runfile_path, "r") as file:
    for line in file:
        if "=" in line:
            key, value = line.strip().split("=", 1)
            key = key.strip()
            value = value.strip()
            try:
                run_config[key] = eval(value)
            except:
                run_config[key] = value


# ------------------------------------------------------------------------------
# Unpack variables
# ------------------------------------------------------------------------------
run_name = run_config["run_name"]
model_name = run_config["model_name"]
dataset_name = run_config["dataset_name"]
target_type = run_config["target_type"]
context_length = run_config["context_length"]

model_base = model_name.split("_model")[0]         # → "chronos"
model_suffix = model_name.split("_model_")[1]      # → "base"
model_variant = f"{model_base}_{model_suffix}"     # → "chronos_base"


# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
runfile_name = runfile_path.stem
output_dir = Path(f"analysis/{model_variant}_{runfile_name}")
input_file = Path(f"forecasts/{model_variant}_{runfile_name}/{dataset_name}_ctx{context_length}.pkl")
series_file = Path(f"datasets/{dataset_name}_{target_type}.csv")

output_dir.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------------
# Load forecast and series
# ------------------------------------------------------------------------------
if not input_file.exists():
    raise FileNotFoundError(f"Forecast result not found: {input_file}")
if not series_file.exists():
    raise FileNotFoundError(f"Series file not found: {series_file}")

with open(input_file, "rb") as f:
    result = pickle.load(f)

series = pd.read_csv(series_file).squeeze()
series.index = range(len(series))

low, median, high, samples, base_price = result
selected_days = [0, 10, 20]
is_price_data = (target_type == "prices")


# ------------------------------------------------------------------------------
# Generate plots
# ------------------------------------------------------------------------------
plot_forecast(
    series=series,
    low=low,
    median=median,
    high=high,
    dgp_type=dataset_name,
    context_length=context_length,
    path=output_dir / f"{dataset_name}_ctx{context_length}_forecast.png",
    is_price_data=is_price_data
)

plot_daily_return_kdes(
    samples=samples,
    selected_days=selected_days,
    dgp_type=dataset_name,
    context_length=context_length,
    path=output_dir / f"{dataset_name}_ctx{context_length}_kdes.png",
    is_price_data=is_price_data
)

plot_daily_return_cdfs(
    samples=samples,
    selected_days=selected_days,
    dgp_type=dataset_name,
    context_length=context_length,
    path=output_dir / f"{dataset_name}_ctx{context_length}_cdfs.png",
    is_price_data=is_price_data
)


# ------------------------------------------------------------------------------
# KL divergence plot
# ------------------------------------------------------------------------------
dgp_path = Path(f"datasets/{dataset_name}_paths.npy")
if not dgp_path.exists():
    print(f"[SKIP] DGP sample paths not found: {dgp_path}")
else:
    dgp_samples = np.load(dgp_path)

    model_returns = samples if target_type == "returns" else samples[:, 1:] / samples[:, :-1] - 1
    dgp_returns = dgp_samples if target_type == "returns" else dgp_samples[:, 1:] / dgp_samples[:, :-1] - 1

    for day in selected_days:
        try:
            kl = compute_kl_divergence(dgp_returns[:, day], model_returns[:, day])
        except Exception as e:
            print(f"[SKIP] KL failed at day {day}: {e}")

    output_kl_plot = output_dir / f"{dataset_name}_ctx{context_length}_kl_plot.png"
    plot_multiple_return_kde_comparison(
        dgp_samples=dgp_returns,
        model_samples=model_returns,
        selected_days=selected_days,
        dgp_type=dataset_name,
        context_length=context_length,
        path=output_kl_plot,
        is_price_data=False  # Already transformed
    )

    print(f"[SAVED] All analysis plots saved in: {output_dir}")
