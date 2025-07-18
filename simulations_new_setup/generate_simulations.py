# ------------------------------------------------------------------------------
# Innosuisse Project: Usability of Transformer Models for Modelling Commodity Markets
# Date: July 16, 2025
# Authors: Peter Gruber, Alessandro Dodon
#
# Script: Generate simulated data (series and paths) based on config YAML
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Packages
# ------------------------------------------------------------------------------
import os
import numpy as np
import argparse
import yaml
from pathlib import Path
from utils.simulations import *
import inspect


# ------------------------------------------------------------------------------
# CLI input
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--sim_file", default="jobs/sim_00.yaml")
args = parser.parse_args()

with open(args.sim_file, "r") as f:
    config = yaml.safe_load(f)

sim_name = Path(args.sim_file).stem


# ------------------------------------------------------------------------------
# Output folders
# ------------------------------------------------------------------------------
base_dir = f"data/{sim_name}"
series_price_dir = f"{base_dir}/prices_simulated_series"
series_return_dir = f"{base_dir}/returns_simulated_series"
paths_price_dir = f"{base_dir}/prices_simulated_paths"
paths_return_dir = f"{base_dir}/returns_simulated_paths"

os.makedirs(series_price_dir, exist_ok=True)
os.makedirs(series_return_dir, exist_ok=True)
os.makedirs(paths_price_dir, exist_ok=True)
os.makedirs(paths_return_dir, exist_ok=True)


# ------------------------------------------------------------------------------
# Run all DGPs
# ------------------------------------------------------------------------------
for dgp_config in config["dgps"]:
    dgp_name = dgp_config["name"]
    dgp_type = dgp_config["type"]
    dgp_params = dgp_config.get("params", {})
    path_params = dgp_config.get("forecast_params", {})
    seed = config["seed"]
    trading_days = config["trading_days"]
    forecast_days = config["forecast_days"]
    n_samples = config["n_samples"]
    initial_price = config["initial_price"]

    print(f"[DGP] Generating: {dgp_name} (type={dgp_type})")

    # Simulate price series
    simulate_func = globals()[f"simulate_{dgp_type}_prices"]
    # Call simulate function with or without seed, depending on its signature
    simulate_signature = inspect.signature(simulate_func)
    if "seed" in simulate_signature.parameters:
        series = simulate_func(trading_days, initial_price, seed=seed, **dgp_params)
    else:
        series = simulate_func(trading_days, initial_price, **dgp_params)


    price_path = f"{series_price_dir}/prices_{dgp_name}_seed{seed}.csv"
    series.to_csv(price_path, index=False, float_format="%.8f")
    print(f"[SAVED] {price_path}")

    # Compute returns
    returns = series.pct_change().dropna().reset_index(drop=True)
    returns_path = f"{series_return_dir}/returns_{dgp_name}_seed{seed}.csv"
    returns.to_csv(returns_path, index=False, float_format="%.8f")
    print(f"[SAVED] {returns_path}")

    # Skip path simulation if not required
    if not dgp_config.get("generate_paths", True):
        continue

    base_price = series.iloc[-1]

    forecast_func = globals()[f"forecast_{dgp_type}_paths"]

    # Special case: dynamic last_return and last_volatility for t_garch
    if dgp_type == "t_garch":
        last_return = returns.iloc[-1]
        last_volatility = dgp_params.get("volatility_start", 0.01)
        path_params = {
            **path_params,
            "last_return": last_return,
            "last_volatility": last_volatility
        }

    # Forecast paths
    paths = forecast_func(
        base_price,
        forecast_days,
        n_samples,
        seed=seed,
        **path_params
    )

    price_paths_file = f"{paths_price_dir}/prices_{dgp_name}_seed{seed}.npy"
    np.save(price_paths_file, paths.astype(np.float64))
    print(f"[SAVED] {price_paths_file}")

    returns_paths = (paths[:, 1:] / paths[:, :-1]) - 1
    returns_paths_file = f"{paths_return_dir}/returns_{dgp_name}_seed{seed}.npy"
    np.save(returns_paths_file, returns_paths.astype(np.float64))
    print(f"[SAVED] {returns_paths_file}")
