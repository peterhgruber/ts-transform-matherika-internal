# ------------------------------------------------------------------------------
# Packages
# ------------------------------------------------------------------------------
import timesfm
import pandas as pd
from scipy.interpolate import interp1d
import numpy as np


# ------------------------------------------------------------------------------
# TimeFM forecast function with inverse transform sampling
# ------------------------------------------------------------------------------
def forecast_timesfm(series, prediction_days=22, context_len=2048, n_samples=1000):
    """
    Forecast future values using TimesFM and reconstruct an empirical distribution
    from multiple quantile levels using inverse transform sampling.

    Returns:
    - low, median, high: 10th, 50th, 90th percentiles
    - samples: simulated forecast paths (n_samples × prediction_days)
    - base_price: last observed value
    """
    df = pd.DataFrame({
        'ds': pd.date_range(start='2020-01-01', periods=len(series), freq='D'),
        'y': series.values,
        'unique_id': 'simulated_series'
    })

    df = df.iloc[-context_len:]

    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="cpu",
            per_core_batch_size=32,
            horizon_len=prediction_days,
            num_layers=50,
            use_positional_embedding=False,
            context_len=context_len
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
        )
    )

    preds_df = tfm.forecast_on_df(
        inputs=df,
        freq="D",
        value_name="y",
        num_jobs=-1
    )

    # Interpolation from quantiles
    quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    samples = np.empty((n_samples, prediction_days))

    for t in range(prediction_days):
        q_values = [preds_df[f'timesfm-q-{q:.1f}'][t] for q in quantile_levels]
        inv_cdf = interp1d(quantile_levels, q_values, kind="linear", fill_value="extrapolate", bounds_error=False)
        u = np.random.uniform(size=n_samples)
        samples[:, t] = inv_cdf(u)

    low, median_out, high = np.quantile(samples, [0.1, 0.5, 0.9], axis=0)
    base_price = series.iloc[-1]

    return low, median_out, high, samples, base_price
