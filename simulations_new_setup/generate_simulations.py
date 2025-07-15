# ------------------------------------------------------------------------------
# Innosuisse Project: Usability of Transformer Models for Modelling Commodity Markets
# Date: July 9, 2025
# Authors: Peter Gruber (peter.gruber@usi.ch), Alessandro Dodon (alessandro.dodon@usi.ch)
#
#  This script defines the data generation process (DGP) for simulating price series
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Packages
# ------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
from utils.sim_utils import *


# ------------------------------------------------------------------------------
# Global parameters
# ------------------------------------------------------------------------------
trading_days = 1000
forecast_days = 22
n_samples = 1000
initial_price = 100.0
seed = 42


# ------------------------------------------------------------------------------
# Output folders
# ------------------------------------------------------------------------------
os.makedirs("data/simulated_prices", exist_ok=True)
os.makedirs("data/simulated_paths", exist_ok=True)


# ------------------------------------------------------------------------------
# Simulation configurations
# ------------------------------------------------------------------------------
dgp_configs = {
    "gbm_low_vol": lambda: simulate_gbm_prices(
        trading_days, initial_price,
        drift=0.0 / trading_days,
        volatility=0.15 / np.sqrt(trading_days),
        seed=seed),

    "gbm_high_vol": lambda: simulate_gbm_prices(
        trading_days, initial_price,
        drift=0.0 / trading_days,
        volatility=0.80 / np.sqrt(trading_days),
        seed=seed),

    "t_garch": lambda: simulate_t_garch_prices(
        trading_days, initial_price,
        omega=0.00001, alpha=0.15, beta=0.8,
        volatility_start=0.01, degrees_freedom=3,
        seed=seed),

    "mixture_normal": lambda: simulate_mixture_normal_prices(
        trading_days, initial_price,
        means=[0.0, -0.002], std_devs=[0.01, 0.03],
        weights=[0.9, 0.1], seed=seed),

    "constant": lambda: simulate_constant_prices(
        trading_days, initial_price),

    "linear": lambda: simulate_linear_prices(
        trading_days, initial_price, daily_return=0.0005),

    "seasonal": lambda: simulate_seasonal_prices(
        trading_days, initial_price,
        amplitude=0.02, frequency=1/60,
        trend=0.00005, noise_std=0.03, seed=seed)
}


# ------------------------------------------------------------------------------
# Run and save all simulations
# ------------------------------------------------------------------------------
for dgp_name, dgp_func in dgp_configs.items():
    series = dgp_func()
    price_path = f"data/simulated_prices/{dgp_name}_seed{seed}.csv"
    series.to_csv(price_path, index=False, float_format="%.8f")
    print(f"Saved: {price_path}")

    base_price = series.iloc[-1]

    if dgp_name in ["constant", "linear"]:
        continue

    if dgp_name == "gbm_low_vol":
        paths = forecast_gbm_paths(base_price, forecast_days, n_samples,
                                   drift=0.0 / trading_days,
                                   volatility=0.15 / np.sqrt(trading_days),
                                   seed=seed)

    elif dgp_name == "gbm_high_vol":
        paths = forecast_gbm_paths(base_price, forecast_days, n_samples,
                                   drift=0.0 / trading_days,
                                   volatility=0.80 / np.sqrt(trading_days),
                                   seed=seed)

    elif dgp_name == "t_garch":
        paths = forecast_t_garch_paths(
            base_price, forecast_days, n_samples,
            omega=0.00001, alpha=0.15, beta=0.8,
            last_volatility=0.01, last_return=0.0,
            degrees_freedom=3, seed=seed)

    elif dgp_name == "mixture_normal":
        paths = forecast_mixture_normal_paths(
            base_price, forecast_days, n_samples,
            means=[0.0, -0.002], std_devs=[0.01, 0.03],
            weights=[0.9, 0.1], seed=seed)

    elif dgp_name == "seasonal":
        paths = forecast_seasonal_paths(
            base_price, forecast_days, n_samples,
            amplitude=0.02, frequency=1/60,
            trend=0.00005, noise_std=0.03, seed=seed)

    else:
        continue

    path_file = f"data/simulated_paths/{dgp_name}_seed{seed}.npy"
    np.save(path_file, paths.astype(np.float64))
    print(f"Saved: {path_file}")
