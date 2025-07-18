# Simulation and Forecasting Framework

This repository provides a complete simulation and forecasting pipeline for evaluating financial models on synthetic data. It allows for easy configuration, execution, and evaluation of multiple model forecasts across varied data-generating processes (DGPs).

---

## Repository Structure

```plaintext
simulations_new_setup/
├── analyze_results.py           # Analyze forecast results: tables, plots, KL
├── generate_simulations.py      # Generate synthetic price/return series
├── run_forecasts.py             # Run model forecasts over simulated data
├── jobs/                        # YAML configuration files
│   ├── sim_00.yaml              # Default simulation setup
│   └── job_00.yaml              # Default forecasting setup
├── data/                        # Generated simulation data (by sim_name)
├── forecasts/                   # Forecast results by model/job/simulation
├── results/                     # Evaluation outputs: tables and plots
│   └── <model_base>_results/
│       └── <model_variant>/
│           └── job_00_sim_00/
│               ├── tables/
│               └── plots/
│                   ├── forecasts/
│                   ├── kdes/
│                   ├── cdfs/
│                   └── kl_divergence/
├── models/                      # Forecasting scripts per model
│   ├── chronos_models/
│   ├── moirai_models/
│   ├── tirex_models/
│   ├── toto_models/
│   ├── timesfm_models/
│   └── lag_llama_models/
├── utils/                       # Common utilities for simulation and plotting
│   ├── simulations.py
│   ├── plotting.py
│   └── evaluation.py
└── README.md
```

---

##  Workflow Overview

### 1. Generate Simulations

Defined by:

```
jobs/sim_00.yaml
```

Run the default simulation job with:

```bash
python generate_simulations.py
```

This creates:

```plaintext
data/sim_00/
├── prices_simulated_series/
├── returns_simulated_series/
├── prices_simulated_paths/
└── returns_simulated_paths/
```

Modify sim_00.yaml to change DGPs, seeds, or volatility assumptions.

### 2. Run Forecasts

The script `run_forecasts.py` allows you to specify which forecast job (`job_file`) and simulation (`sim_name`) to use.

By default, it uses:

- `jobs/job_00.yaml` for forecasting parameters  
- `data/sim_00/` for simulation data  
- Forecast target type: `"prices"`

You can override all of these via command-line flags.

#### Basic usage

This runs the default forecast setup:

```bash
python run_forecasts.py --model_name chronos_model_base --target_type prices
```

#### Custom job and simulation

Specify an alternative YAML file and simulation name like this:

```bash
python run_forecasts.py \
  --model_name chronos_model_base \
  --target_type returns \
  --job_file jobs/job_custom.yaml \
  --sim_name sim_custom
```

This will forecast for:
- Job: `jobs/job_custom.yaml`
- Simulation: `data/sim_custom/`
- Target type: returns

#### Default job_00.yaml settings

When using the default job file `job_00.yaml`, the forecast loop runs with:

- Forecast horizon: 22 days
- Context lengths: [22, 66, 252]
- Forecast samples: 1000
- Seed: 42
- DGPs:
  - `gbm_low_vol`
  - `gbm_high_vol`
  - `t_garch`
  - `mixture_normal`
  - `constant`
  - `linear`
  - `seasonal`

#### Output structure

Forecast results are saved to:

```plaintext
forecasts/<model_base>/<model_variant>/<job_name>_<sim_name>/<prices|returns>/
```

Example:

```plaintext
forecasts/chronos/base/job_00_sim_00/prices/
```

Each file is named:

```plaintext
<dgp_name>_ctx<context_length>_seed<seed>.pkl
```

Example:

```plaintext
mixture_normal_ctx66_seed42.pkl
```

### 3. Analyze Forecasts

You can analyze any forecast job and simulation by specifying:

- `--model_name`: required  
- `--target_type`: `"prices"` or `"returns"`  
- `--job_file`: YAML file used for the forecasting job  
- `--sim_name`: which simulation to analyze (matches the folder under `data/`)

#### Default usage

This analyzes the default job `job_00.yaml` and simulation `sim_00`:

```bash
python analyze_results.py --model_name chronos_model_base --target_type prices
```

#### Custom job and simulation

Specify a different forecast job and simulation:

```bash
python analyze_results.py \
  --model_name chronos_model_base \
  --target_type returns \
  --job_file jobs/job_custom.yaml \
  --sim_name sim_custom
```

This will read forecast data from:

```plaintext
forecasts/<model_base>/<model_variant>/job_custom_sim_custom/<returns|prices>/
```

and save analysis output to:

```plaintext
results/<model_base>_results/<model_variant>/job_custom_sim_custom/<returns|prices>/
```

Generated outputs include:

- Forecast summary table (`forecast_table.tex`)
- Percentile table (`percentiles_table.tex`)
- KL divergence table (`kl_divergence_table.tex`)
- Plots:
  - Forecast paths
  - Return KDEs and CDFs
  - Return density comparisons via KL divergence

---

## Supported Models

The following models are implemented in the `models/` folder and can be used via the `--model_name` flag:

- `chronos_model_tiny`
- `chronos_model_miny`
- `chronos_model_base`
- `chronos_model_large` †
- `moirai_model_small`
- `moirai_model_base`
- `moirai_model_large` †
- `tirex_model`
- `toto_model`
- `timesfm_model_small` *
- `timesfm_model_large` *
- `lag_llama_model`

> Models marked with * require special setup. See note below for TimeSFM.
> Models marked with † are implemented but not yet tested in forecast runs

---

## Note on TimeSFM Setup

TimeSFM requires a dedicated Python environment due to specific PyTorch and dependency requirements.

### Environment Setup

Make sure to use a clean environment with:

- **Python 3.11** (ideally Python 3.11.11)
- Installed via `venv` or `conda`

Example:

```bash
conda create -n timesfm-env python=3.11
conda activate timesfm-env
pip install torch pandas numpy scipy timesfm
```

> We created a Jupyter kernel named:
> **"Python 3.11 (timesfm-env) (Python 3.11.11)"**

### Running TimeSFM

Use the full path to the correct Python executable when calling the forecast script:

```bash
/Users/aledo/timesfm-env/bin/python run_forecasts.py --model_name timesfm_model_large --target_type prices
```

Then analyze results normally:

```bash
python analyze_results.py --model_name timesfm_model_large --target_type prices
```
