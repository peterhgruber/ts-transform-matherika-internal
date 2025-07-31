# Instructions

This repository provides a framework for running and evaluating various time series forecasting models. Below are the setup steps to ensure all models run in properly configured environments.

## Environment Setup

This repository uses two separate Conda environments to cleanly separate dependencies and avoid package conflicts across models.

Inside the `envs/` folder, you will find the following full environment specification files:

- `timesfm_environment_full.yml`: for TimesFM models (requires Python 3.11 and PyTorch backend)
- `tsfm_environment_full.yml`: for all other time series forecasting models

To create the environments, run the following commands:

```bash
conda env create -f envs/tsfm_environment_full.yml
conda env create -f envs/timesfm_environment_full.yml
```

Once created, you can activate them as needed:

```bash
conda activate tsfm
conda activate timesfm
```

## Generating Simulations

To generate synthetic time series data from different data-generating processes (DGPs), open the notebook:

`simulate_dgps.ipynb`

Run all cells to generate the simulations.  
Parameters such as the DGP type, number of paths, or time horizon can be modified directly in the notebook to suit your needs.

## Creating Runfiles

Before running forecasts, you must first generate the runfiles that specify which models will be used on which DGPs.

To do this, run the script:

`generate_runfiles.py`

This script lets you select:

- The models to run
- The DGPs to forecast
- Target type (prices or returns)
- Context lengths and prediction horizons

A numeric selection menu is used to configure these options.  
Once configured, the script will automatically generate a set of `.txt` runfiles in the `runfiles/` directory.

This is faster than manual creation, which is still an option.


## Running Forecasts

There are two options to run forecasts based on the runfiles you previously generated:

### Option 1: Run a Single Forecast

Use the `run_forecast.py` script to run a single job specified by a `.txt` runfile.

From the terminal, run:

```bash
python run_forecast.py runfiles/<your_runfile>.txt
```

This will:
- Load the model and forecast function
- Load the corresponding dataset
- Run the forecast
- Save the output in forecasts/<run_name>.pkl
Be sure to activate the appropriate Conda environment (tsfm or timesfm) before running the script, depending on the model used in the runfile.

### Option 2: Run All Forecasts Automatically

To run all forecasts at once using the prepared runfiles, use the provided runner scripts.

For all models **except** TimesFM, activate the `tsfm` environment and run:

```bash
conda activate tsfm
python run_all_forecasts.py
```

This script scans the `runfiles/` folder, loads the appropriate forecasting function for each model, runs the forecasts, and saves the results to the forecasts/ directory.
For the TimesFM model, activate the timesfm environment and run:

```bash
conda activate timesfm
python run_all_forecasts_timesfm.py
```

This version handles only TimesFM models and behaves similarly, processing each runfile automatically and storing outputs in the same `forecasts/` directory.

## Analyzing Results

Once forecasts have been generated and saved, you can analyze the results using the provided Jupyter notebooks.

These notebooks load the forecast files from the `forecasts/` folder and compute various summary statistics and comparison metrics.

You can modify the parameters in each notebook to:
- Select which models to include
- Choose which DGPs or context lengths to analyze
- Switch between returns and prices

Currently, the repository includes:

- `q0_summary.ipynb`: provides an overall summary of forecast performance across models and setups.
- `q1_*.ipynb`, `q2_*.ipynb`, `q3_*.ipynb`, `q4_*.ipynb`: each addresses a specific analysis question in more depth.
