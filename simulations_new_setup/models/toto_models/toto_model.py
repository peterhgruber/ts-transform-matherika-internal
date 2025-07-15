# ------------------------------------------------------------------------------
# Innosuisse Project: Usability of Transformer Models for Modelling Commodity Markets
# Date: June 29, 2025
# Authors: Peter Gruber, Alessandro Dodon
#
# This script defines the Toto forecasting function using the official open model.
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Packages
# ------------------------------------------------------------------------------
import os
import sys
import subprocess
from pathlib import Path
import torch
import numpy as np
import pandas as pd


# ------------------------------------------------------------------------------
# Setup: Clone and prepare the Toto repo in tmp/
# ------------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
WORKING_DIR = SCRIPT_DIR / ".." / ".." / "tmp" / "toto"
REPO_PATH = WORKING_DIR / "toto_github"

os.makedirs(WORKING_DIR, exist_ok=True)

# Clone the repo if missing
if not REPO_PATH.exists():
    print("Cloning Toto repo into:", REPO_PATH)
    subprocess.run(
        ["git", "clone", "https://github.com/DataDog/toto.git", str(REPO_PATH)],
        check=True
    )

# Add Toto repo root to sys.path for relative imports to work
if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))


# ------------------------------------------------------------------------------
# Import Toto modules
# ------------------------------------------------------------------------------
from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto


# ------------------------------------------------------------------------------
# Load Toto model once globally
# ------------------------------------------------------------------------------
_toto_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_toto_model = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0").to(_toto_device)
_toto_model.compile()
_forecaster = TotoForecaster(_toto_model.model)


# ------------------------------------------------------------------------------
# Forecast function
# ------------------------------------------------------------------------------
def forecast_toto(series, context_length=2048, prediction_days=336, n_samples=256,
                  device=None, pipeline=None):
    """
    Forecast future paths using the Toto foundation model (univariate only).

    Returns:
    - low, median, high: np.ndarrays of shape (prediction_length,)
    - forecasted_paths: np.ndarray of shape (n_samples, prediction_length)
    - last_observed_value: float
    """
    if isinstance(series, pd.Series):
        series = torch.tensor(series.values, dtype=torch.float32)

    if series.ndim != 1:
        raise ValueError("Input series must be 1D (univariate).")

    if device is None:
        device = _toto_device

    input_tensor = series[-context_length:].unsqueeze(0).to(device)
    timestamps = torch.zeros_like(input_tensor)
    intervals = torch.full((1,), 60 * 15, device=device)

    inputs = MaskedTimeseries(
        series=input_tensor,
        padding_mask=torch.full_like(input_tensor, True, dtype=torch.bool),
        id_mask=torch.zeros_like(input_tensor),
        timestamp_seconds=timestamps,
        time_interval_seconds=intervals,
    )

    forecast = _forecaster.forecast(
        inputs,
        prediction_length=prediction_days,
        num_samples=n_samples,
        samples_per_batch=n_samples,
    )

    forecasted_paths = np.asarray(forecast.samples).squeeze().T  # shape: (n_samples, prediction_days)
    low, median, high = np.quantile(forecasted_paths, [0.1, 0.5, 0.9], axis=0)
    last_observed_value = series[-1].item()

    return low, median, high, forecasted_paths, last_observed_value
