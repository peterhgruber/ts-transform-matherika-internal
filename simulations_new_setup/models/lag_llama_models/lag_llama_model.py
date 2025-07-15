# ------------------------------------------------------------------------------
# Innosuisse Project: Usability of Transformer Models for Modelling Commodity Markets
# Date: June 29, 2025
# Authors: Peter Gruber, Alessandro Dodon
#
# This script loads Lag-Llama and defines the simple forecasting function.
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------
import os
import sys
import subprocess
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
WORKING_DIR = SCRIPT_DIR / ".." / ".." / "tmp" / "lag_llama"
REPO_PATH = WORKING_DIR / "lag-llama"
HF_CACHE = WORKING_DIR / "huggingface"

# Add to sys.path
sys.path.append(str(REPO_PATH))
os.makedirs(WORKING_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# Clone repo if missing
# ------------------------------------------------------------------------------
if not REPO_PATH.exists():
    subprocess.run(
        ["git", "clone", "https://github.com/time-series-foundation-models/lag-llama/"],
        cwd=WORKING_DIR,
        check=True,
    )

# ------------------------------------------------------------------------------
# Install dependencies if not done already
# ------------------------------------------------------------------------------
requirements_marker = REPO_PATH / ".requirements_installed"
if not requirements_marker.exists():
    subprocess.run(["pip", "install", "-r", "requirements.txt"], cwd=REPO_PATH, check=True)
    requirements_marker.write_text("installed")


# ------------------------------------------------------------------------------
# Download checkpoint
# ------------------------------------------------------------------------------
os.environ["HF_HOME"] = str(HF_CACHE)
subprocess.run([
    "huggingface-cli", "download",
    "time-series-foundation-models/Lag-Llama", "lag-llama.ckpt",
    "--local-dir", str(REPO_PATH),
], check=True)


# ------------------------------------------------------------------------------
# Final imports
# ------------------------------------------------------------------------------
from gluonts.dataset.pandas import PandasDataset
from gluonts.torch.distributions.studentT import StudentTOutput
from gluonts.torch.modules.loss import NegativeLogLikelihood
from lag_llama.gluon.estimator import LagLlamaEstimator

print("Lag-Llama setup complete and ready to forecast.")


# ------------------------------------------------------------------------------
# Forecast function
# ------------------------------------------------------------------------------
def forecast_lag_llama(series, context_length=66, prediction_days=22, n_samples=1000, device=None):
    """
    Forecast future price paths using the Lag-Llama time series foundation model.
    """

    torch.serialization.add_safe_globals([StudentTOutput, NegativeLogLikelihood])
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Format input series
    df = pd.DataFrame({
        "Date": pd.date_range(start="2020-01-01", periods=len(series), freq="D"),
        "Value": series.astype("float32").values,
        "ID": "Sim"
    })

    dataset = PandasDataset.from_long_dataframe(
        dataframe=df,
        target="Value",
        item_id="ID",
        timestamp="Date",
        freq="D"
    )

    # Load model
    ckpt_path = REPO_PATH / "lag-llama.ckpt"
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    args = ckpt["hyper_parameters"]["model_kwargs"]

    estimator = LagLlamaEstimator(
        ckpt_path=str(ckpt_path),
        prediction_length=prediction_days,
        context_length=context_length,
        input_size=args["input_size"],
        n_layer=args["n_layer"],
        n_embd_per_head=args["n_embd_per_head"],
        n_head=args["n_head"],
        scaling=args["scaling"],
        time_feat=args["time_feat"],
        batch_size=1,
        num_parallel_samples=n_samples,
        device=device,
    )

    predictor = estimator.create_predictor(
        transformation=estimator.create_transformation(),
        module=estimator.create_lightning_module()
    )

    forecasts = list(predictor.predict(dataset=dataset))
    samples = forecasts[0].samples  # shape: (n_samples, prediction_days)

    low, median, high = np.quantile(samples, [0.1, 0.5, 0.9], axis=0)
    base_price = series.iloc[-1]

    return low, median, high, samples, base_price
