# ------------------------------------------------------------------------------
# Packages
# ------------------------------------------------------------------------------
import numpy as np
import torch
import timesfm


# ------------------------------------------------------------------------------
# TimesFM 2.5 forecast function
# ------------------------------------------------------------------------------
def forecast_timesfm(series, prediction_days=22, context_len=2048, n_samples=1000):
    """
    Forecast future values using TimesFM 2.5 (200M parameters).

    Returns:
    - low, median, high: 10th, 50th, 90th percentiles
    - samples: simulated forecast paths (n_samples × prediction_days)
    - base_price: last observed value
    """
    context_array = series[-context_len:].values.astype(np.float32)
    base_price = series.iloc[-1]

    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch", torch_compile=True
    )

    model.compile(
        timesfm.ForecastConfig(
            max_context=context_len,
            max_horizon=prediction_days,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )

    point_forecast, quantile_forecast = model.forecast(
        horizon=prediction_days,
        inputs=[context_array],
    )

    q = quantile_forecast[0]
    low = q[:, 0]
    median_out = q[:, 5]
    high = q[:, 9]

    quantile_levels = np.linspace(0.1, 0.9, q.shape[-1])
    samples = np.empty((n_samples, prediction_days))
    for t in range(prediction_days):
        inv_cdf = np.interp(
            np.random.uniform(size=n_samples),
            quantile_levels,
            q[t]
        )
        samples[:, t] = inv_cdf

    return low, median_out, high, samples, base_price