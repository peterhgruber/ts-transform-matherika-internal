# ------------------------------------------------------------------------------
# Innosuisse Project: Usability of Transformer Models for Modelling Commodity Markets
# Date: June 29, 2025
# Authors: Peter Gruber (peter.gruber@usi.ch), Alessandro Dodon (alessandro.dodon@usi.ch)
#
# This script sets up TiRex and defines the forecasting function.
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Packages
# ------------------------------------------------------------------------------
import os
import sys
import torch
import numpy as np
from scipy.interpolate import interp1d
from tirex import load_model

# Set CUDA_LIB to avoid KeyError from TiRex internals
os.environ.setdefault("CUDA_LIB", "")
os.environ.setdefault("TIREX_NO_CUDA", "1")  # Force CPU by default


# ------------------------------------------------------------------------------
# TiRex forecast function with inverse transform sampling
# ------------------------------------------------------------------------------
def forecast_tirex(series, context_length=66, prediction_days=None, n_samples=1000,
                   selected_days=None, device=None):
    """
    Forecast future values using TiRex and reconstruct an empirical distribution
    from multiple quantile levels using inverse transform sampling.

    Returns:
    - low, median, high: 10th, 50th, 90th percentiles
    - samples: simulated forecast paths (n_samples × prediction_days)
    - base_price: last observed value
    """
    if selected_days is None:
        selected_days = [0, 10, 20]  # or pull from a global config

    required_min_forecast = max(selected_days) + 2
    if prediction_days is None or prediction_days < required_min_forecast:
        prediction_days = required_min_forecast

    if device is None or not torch.cuda.is_available():
        device = torch.device("cpu")

    model = load_model("NX-AI/TiRex", device=device)

    clean_series = series.fillna(method='ffill').fillna(method='bfill')
    context_array = clean_series[-context_length:].values.astype(np.float32)
    context_tensor = torch.tensor(context_array, dtype=torch.float32).unsqueeze(0)
    context_tensor_3d = context_tensor.unsqueeze(1)  # shape (1, 1, context_length)

    quantiles, means = model.forecast(
        context=[context_tensor_3d.squeeze()],
        prediction_length=prediction_days,
        output_type="numpy"
    )

    quantiles = quantiles[0]  # shape (prediction_days, 9)
    quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    samples = np.empty((n_samples, prediction_days))
    for t in range(prediction_days):
        q_values = quantiles[t, :]
        inv_cdf = interp1d(quantile_levels, q_values, kind="linear", fill_value="extrapolate", bounds_error=False)
        u = np.random.uniform(size=n_samples)
        samples[:, t] = inv_cdf(u)

    low = quantiles[:, 0]
    median = quantiles[:, 4]
    high = quantiles[:, -1]
    base_price = series.iloc[-1]

    return low, median, high, samples, base_price
