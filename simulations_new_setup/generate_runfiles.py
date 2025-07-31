# ------------------------------------------------------------------------------
# Packages
# ------------------------------------------------------------------------------
from pathlib import Path


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
model_names = [
    "chronos_model_tiny",
    "chronos_model_mini",
    "chronos_model_base",
    "lag_llama_model",
    "moirai_model_small",
    "moirai_model_base",
    "toto_model",
    "tirex_model",
    "timesfm_model_small",
    "timesfm_model_large"
]

dgp_names = [
    "gbm_low_vol",
    "gbm_high_vol",
    "t_garch",
    "mixture_normal",
    "seasonal"
]

context_lengths = [22, 66, 252]
prediction_days = 22
target_types = ["prices", "returns"]

forecast_samples = 1000
output_folder = Path("runfiles")
output_folder.mkdir(exist_ok=True)


# ------------------------------------------------------------------------------
# Runfile generator
# ------------------------------------------------------------------------------
counter = 1

for model in model_names:
    for dgp in dgp_names:
        for context_length in context_lengths:
            for target_type in target_types:

                runfile_text = f"""run_name = forecast_{counter:03}
model_name = {model}
dataset_name = {dgp}
target_type = {target_type}
context_length = {context_length}
prediction_days = {prediction_days}
forecast_samples = {forecast_samples}
"""

                runfile_path = output_folder / f"forecast_{counter:03}.txt"
                with open(runfile_path, "w") as f:
                    f.write(runfile_text)

                counter += 1

print(f"[DONE] Generated {counter - 1} runfiles in: {output_folder}")
