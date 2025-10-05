# ------------------------------------------------------------------------------
# Packages
# ------------------------------------------------------------------------------
from pathlib import Path
import re


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
new_model_names = [
    "timesfm_model_2_5", # Only keep TimesFM 2.5
]

dgp_names = [
    "gbm_low_vol",
    "gbm_high_vol",
    "garch",
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
# Determine the next counter index
# ------------------------------------------------------------------------------
existing_files = list(output_folder.glob("forecast_*.txt"))
if existing_files:
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
# Generate new TimesFM 2.5 runfiles
# ------------------------------------------------------------------------------
for my_model_name in new_model_names:
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

                if runfile_path.exists():
                    skipped_count += 1
                else:
                    with open(runfile_path, "w") as my_file:
                        my_file.write(runfile_text)
                    created_count += 1

                counter += 1

print(f"[DONE] Added {created_count} new TimesFM 2.5 runfiles, skipped {skipped_count} existing.")
