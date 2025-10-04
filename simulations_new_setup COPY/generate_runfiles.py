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
    "moirai_model_small_2_0",  # added
    "moirai_model_small_1_1",  # added
    "moirai_model_base_1_1",   # added
    "toto_model",
    "tirex_model",
    "timesfm_model_small",
    "timesfm_model_large",
]

dgp_names = [
    "gbm_low_vol",
    "gbm_high_vol",
    "garch",        # added
    "t_garch",
    "mixture_normal",
    "seasonal",
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
created_count = 0
skipped_count = 0

for my_model_name in model_names:
    for my_dgp_name in dgp_names:
        for my_context_length in context_lengths:
            for my_target_type in target_types:

                runfile_text = f"""run_name = forecast_{counter:03}
model_name = {my_model_name}
dataset_name = {my_dgp_name}
target_type = {my_target_type}
context_length = {my_context_length}
prediction_days = {prediction_days}
forecast_samples = {forecast_samples}
"""

                runfile_path = output_folder / f"forecast_{counter:03}.txt"

                # Check if the corresponding file already exists in output directory.
                # If it exists, skip generating this specific file.
                if runfile_path.exists():
                    skipped_count += 1
                else:
                    with open(runfile_path, "w") as my_file:
                        my_file.write(runfile_text)
                    created_count += 1

                counter += 1

print(f"[DONE] Created {created_count} new runfiles, skipped {skipped_count} existing, in: {output_folder}")