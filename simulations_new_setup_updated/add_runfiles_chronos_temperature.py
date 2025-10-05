# ------------------------------------------------------------------------------
# Packages
# ------------------------------------------------------------------------------
from pathlib import Path
import re


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
chronos_models = [
    "chronos_model_tiny",
    "chronos_model_small",
    "chronos_model_base",
]

dgp_names = [
    "gbm_high_vol",
    "t_garch",
    "mixture_normal",
]

context_lengths = [22, 66, 252]
prediction_days = 22
target_types = ["prices", "returns"]
forecast_samples = 1000

# Temperature grid
temperature_grid = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]

output_folder = Path("runfiles")
output_folder.mkdir(exist_ok=True)


# ------------------------------------------------------------------------------
# Determine the next counter index
# ------------------------------------------------------------------------------
existing_files = list(output_folder.glob("forecast_*.txt"))
if existing_files:
    import re
    existing_numbers = [
        int(re.search(r"forecast_(\d+)\.txt", f.name).group(1))
        for f in existing_files
    ]
    counter = max(existing_numbers) + 1
else:
    counter = 1

created_count = 0
skipped_count = 0


# ------------------------------------------------------------------------------
# Generate new Chronos temperature runfiles
# ------------------------------------------------------------------------------
for model_name in chronos_models:
    for dgp_name in dgp_names:
        for context_length in context_lengths:
            for target_type in target_types:
                for temp in temperature_grid:

                    runfile_text = f"""run_name = forecast_{counter:03}
model_name = {model_name}
dataset_name = {dgp_name}
target_type = {target_type}
context_length = {context_length}
prediction_days = {prediction_days}
forecast_samples = {forecast_samples}
temperature = {temp}
"""

                    runfile_path = output_folder / f"forecast_{counter:03}.txt"

                    if runfile_path.exists():
                        skipped_count += 1
                    else:
                        with open(runfile_path, "w") as file:
                            file.write(runfile_text)
                        created_count += 1

                    counter += 1

print(f"[DONE] Created {created_count} new Chronos temperature runfiles, skipped {skipped_count} existing.")