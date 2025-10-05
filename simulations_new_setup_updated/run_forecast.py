# ------------------------------------------------------------------------------
# Packages
# ------------------------------------------------------------------------------
import importlib
import inspect
import pickle
import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path


# ------------------------------------------------------------------------------
# Read .txt runfile
# ------------------------------------------------------------------------------
if len(sys.argv) < 2:
    raise ValueError("Usage: python run_forecast.py <runfile.txt>")

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
# Unpack config variables
# ------------------------------------------------------------------------------
run_name = run_config["run_name"]
model_name = run_config["model_name"]
dataset_name = run_config["dataset_name"]
target_type = run_config["target_type"]
context_length = run_config["context_length"]
prediction_days = run_config["prediction_days"]
forecast_samples = run_config["forecast_samples"]

model_base = model_name.split("_model")[0]         
if "_model_" in model_name:
    model_base, model_suffix = model_name.split("_model_")
    model_variant = f"{model_base}_{model_suffix}"
else:
    model_base = model_name.split("_model")[0]
    model_variant = model_base


# ------------------------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------------------------
series_path = Path(f"datasets/{dataset_name}_{target_type}.csv")
if not series_path.exists():
    raise FileNotFoundError(f"Series file not found: {series_path}")

series = pd.read_csv(series_path).squeeze()
series.index = range(len(series))


# ------------------------------------------------------------------------------
# Load model and forecast function
# ------------------------------------------------------------------------------
model_module = importlib.import_module(f"models.{model_base}_models.{model_name}")

forecast_function = None
for name in dir(model_module):
    if name.startswith("forecast_") and callable(getattr(model_module, name)):
        forecast_function = getattr(model_module, name)
        break
if forecast_function is None:
    raise ValueError("No forecast_* function found in the model module.")

pipeline = getattr(model_module, "pipeline", None)
forecast_params = inspect.signature(forecast_function).parameters


# ------------------------------------------------------------------------------
# Prepare forecast inputs
# ------------------------------------------------------------------------------
args = {
    "series": series,
    "context_length": context_length,
    "prediction_days": prediction_days,
    "n_samples": forecast_samples,
    "pipeline": pipeline if "pipeline" in forecast_params else None,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu") if "device" in forecast_params else None
}

filtered_args = {k: v for k, v in args.items() if v is not None and k in forecast_params}


# ------------------------------------------------------------------------------
# Run forecast
# ------------------------------------------------------------------------------
print(f"[RUN] {run_name} | Model: {model_name} | Dataset: {dataset_name} | Context: {context_length}")
result = forecast_function(**filtered_args)

# Round result
def round_result(obj, decimals=8):
    if isinstance(obj, np.ndarray) and np.issubdtype(obj.dtype, np.floating):
        return np.round(obj.astype(np.float64), decimals)
    elif isinstance(obj, list):
        return [round_result(x, decimals) for x in obj]
    elif isinstance(obj, dict):
        return {k: round_result(v, decimals) for k, v in obj.items()}
    else:
        return obj

result = round_result(result)


# ------------------------------------------------------------------------------
# Save output
# ------------------------------------------------------------------------------
runfile_name = runfile_path.stem
output_file = Path(f"forecasts/{run_name}.pkl")
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, "wb") as f:
    pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"[SAVED] Forecast saved in: {output_file}")
