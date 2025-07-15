# ------------------------------------------------------------------------------
# Innosuisse Project: Usability of Transformer Models for Modelling Commodity Markets
# Date: July 9, 2025
# Authors: Peter Gruber (peter.gruber@usi.ch), Alessandro Dodon (alessandro.dodon@usi.ch)
#
# This script defines the Chronos MLX pipeline and a simple forecasting function
# optimized for Apple Silicon using the experimental MLX backend.
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Packages
# ------------------------------------------------------------------------------
import numpy as np
from chronos_mlx import ChronosPipeline


# ------------------------------------------------------------------------------
# Load the Chronos pipeline (MLX version for Apple Silicon)
# ------------------------------------------------------------------------------
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-tiny",  # other sizes: chronos-t5-mini, -base, -large
    dtype="bfloat16",           # required for MLX
)


# ------------------------------------------------------------------------------
# Chronos forecast function using sequential index (no dates)
# ------------------------------------------------------------------------------
def forecast_chronos(series, pipeline, context_length=66, prediction_days=22,
                     n_samples=100, temperature=1.0, top_k=50, top_p=1.0):
    """
    Forecast future paths from a simulated series using Chronos (MLX version).

    Parameters:
    - series: pd.Series, the simulated price series
    - pipeline: ChronosPipeline (MLX)
    - context_length: number of past points to condition on
    - prediction_days: number of steps ahead to forecast
    - n_samples: number of trajectories to sample
    - temperature, top_k, top_p: sampling controls

    Returns:
    - low, median, high: 10-50-90 percentiles (np.ndarrays)
    - samples: all forecast paths
    - base_price: last actual value used for forecasting
    """
    context_array = series[-context_length:].values

    forecasts = pipeline.predict(
        context=context_array,
        prediction_length=prediction_days,
        num_samples=n_samples,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    samples = forecasts[0]  # shape: [n_samples, prediction_days]
    low, median, high = np.quantile(samples, [0.1, 0.5, 0.9], axis=0)

    return low, median, high, samples, context_array[-1]
