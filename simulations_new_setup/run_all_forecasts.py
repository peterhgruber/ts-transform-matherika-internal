# ------------------------------------------------------------------------------
# Packages
# ------------------------------------------------------------------------------
import importlib
import inspect
import pickle
import numpy as np
import pandas as pd
import torch
import warnings
import logging
from pathlib import Path
from collections import defaultdict


# ------------------------------------------------------------------------------
# Config logging and warnings
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[LOG] %(message)s")
warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    runfile_folder = Path("runfiles")
    runfiles = sorted(runfile_folder.glob("forecast_*.txt"))
    logging.info(f"Found {len(runfiles)} runfiles.")

    grouped_runfiles = defaultdict(list)

    # Group all except TimesFM
    for runfile_path in runfiles:
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

        model_name = run_config.get("model_name", "")
        if model_name.startswith("timesfm_model"):
            continue  # Skip TimesFM
        grouped_runfiles[model_name].append((runfile_path, run_config))

    # Run models
    for model_name, jobs in grouped_runfiles.items():
        logging.info(f"\n[MODEL] {model_name} ({len(jobs)} jobs)")

        if "_model_" in model_name:
            model_base, model_suffix = model_name.split("_model_")
            model_variant = f"{model_base}_{model_suffix}"
        else:
            model_base = model_name.split("_model")[0]
            model_variant = model_base

        try:
            model_module = importlib.import_module(f"models.{model_base}_models.{model_name}")
        except Exception as e:
            logging.error(f"Failed to import module for {model_name}: {e}")
            continue

        forecast_function = None
        for name in dir(model_module):
            if name.startswith("forecast_") and callable(getattr(model_module, name)):
                forecast_function = getattr(model_module, name)
                break
        if forecast_function is None:
            logging.error(f"No forecast_* function found in {model_name}")
            continue

        pipeline = getattr(model_module, "pipeline", None)
        forecast_params = inspect.signature(forecast_function).parameters

        for runfile_path, run_config in jobs:
            run_name = run_config["run_name"]
            dataset_name = run_config["dataset_name"]
            target_type = run_config["target_type"]
            context_length = run_config["context_length"]
            prediction_days = run_config["prediction_days"]
            forecast_samples = run_config["forecast_samples"]

            logging.info(f"[RUN] {runfile_path.name} | Dataset: {dataset_name} | Context: {context_length}")

            series_path = Path(f"datasets/{dataset_name}_{target_type}.csv")
            if not series_path.exists():
                logging.warning(f"[SKIP] Series not found: {series_path}")
                continue

            series = pd.read_csv(series_path).squeeze()
            series.index = range(len(series))

            args = {
                "series": series,
                "context_length": context_length,
                "prediction_days": prediction_days,
                "n_samples": forecast_samples,
                "pipeline": pipeline if "pipeline" in forecast_params else None,
                "device": torch.device("cuda" if torch.cuda.is_available() else "cpu") if "device" in forecast_params else None
            }
            filtered_args = {k: v for k, v in args.items() if v is not None and k in forecast_params}

            try:
                result = forecast_function(**filtered_args)
            except Exception as e:
                logging.error(f"[ERROR] Forecast failed for {run_name}: {e}")
                continue

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

            output_file = Path(f"forecasts/{run_name}.pkl")
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "wb") as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

            logging.info(f"[SAVED] → {output_file}")


# ------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
