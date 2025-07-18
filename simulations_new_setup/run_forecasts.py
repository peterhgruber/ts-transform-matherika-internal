# ------------------------------------------------------------------------------
# Innosuisse Project: Usability of Transformer Models for Modelling Commodity Markets
# Date: July 16, 2025
# Authors: Peter Gruber, Alessandro Dodon
#
# Script: Generic forecast runner for any model 
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Packages
# ------------------------------------------------------------------------------
import argparse
import yaml
import pandas as pd
import pickle
import os
import importlib
from pathlib import Path
import inspect
import torch
import numpy as np


# ------------------------------------------------------------------------------
# CLI input
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", required=True)
parser.add_argument("--job_file", default="jobs/job_00.yaml")
parser.add_argument("--target_type", choices=["prices", "returns"], default="prices")
parser.add_argument("--sim_name", default="sim_00")

args = parser.parse_args()

job_file_path = Path(args.job_file)
job_name = job_file_path.stem  

model_name = args.model_name
model_base = model_name.split("_model")[0]
model_variant = model_name.replace(f"{model_base}_model_", "")

target_type = args.target_type


# ------------------------------------------------------------------------------
# Load job config
# ------------------------------------------------------------------------------
with open(args.job_file, "r") as f:
    config = yaml.safe_load(f)

dgp_list = config["dgp_list"]
context_lengths = config["context_lengths"]
n_samples = config["forecast_samples"]
prediction_days = config["prediction_days"]
seed = config["seed"]


# ------------------------------------------------------------------------------
# Load model 
# ------------------------------------------------------------------------------
model_module = importlib.import_module(f"models.{model_base}_models.{model_name}")

# Detect forecast function automatically
forecast_function = None
for name in dir(model_module):
    if name.startswith("forecast_") and callable(getattr(model_module, name)):
        forecast_function = getattr(model_module, name)
        break
if forecast_function is None:
    raise ValueError("No forecast_* function found in the model module.")

# Try to get pipeline, fallback to None
pipeline = getattr(model_module, "pipeline", None)

# Inspect forecast function parameters
forecast_params = inspect.signature(forecast_function).parameters


# ------------------------------------------------------------------------------
# Forecast loop
# ------------------------------------------------------------------------------
def main():
    for dgp in dgp_list:
        input_path = f"data/{args.sim_name}/{target_type}_simulated_series/{target_type}_{dgp}_seed{seed}.csv"
        if not os.path.exists(input_path):
            print(f"[SKIP] Missing price file: {input_path}")
            continue

        series = pd.read_csv(input_path).squeeze()
        series.index = range(len(series))

        for context in context_lengths:
            print(f"[RUN] DGP: {dgp} | Context: {context}")

            # Prepare arguments
            arguments = {
                "series": series,
                "pipeline": pipeline,
                "context_length": context,
                "prediction_days": prediction_days,
                "n_samples": n_samples
            }

            # If forecast function accepts 'device', detect it and add it
            if "device" in forecast_params:
                arguments["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Filter arguments based on the function signature
            filtered_args = {
                key: val for key, val in arguments.items()
                if key in forecast_params
            }

            # Call forecast function
            result = forecast_function(**filtered_args)

            # Convert result to float64 and round to 8 decimal digits
            def round_forecast_result(obj, decimals=8):
                if isinstance(obj, np.ndarray) and np.issubdtype(obj.dtype, np.floating):
                    return np.round(obj.astype(np.float64), decimals)
                elif isinstance(obj, list):
                    return [round_forecast_result(x, decimals) for x in obj]
                elif isinstance(obj, dict):
                    return {k: round_forecast_result(v, decimals) for k, v in obj.items()}
                else:
                    return obj

            result = round_forecast_result(result)

            # Save results
            output_dir = Path(f"forecasts/{model_base}/{model_variant}/{job_name}_{args.sim_name}/{target_type}")
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / f"{dgp}_ctx{context}_seed{seed}.pkl"
            with open(output_path, "wb") as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"[SAVED] {output_path}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
