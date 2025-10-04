# ------------------------------------------------------------------------------
# Packages
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from gluonts.dataset.pandas import PandasDataset
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module


# ------------------------------------------------------------------------------
# Moirai-2.0 forecast function (small) with inverse-CDF sampling from quantiles
# ------------------------------------------------------------------------------
def forecast_moirai(series, context_length=66, prediction_days=22, n_samples=100):
    """
    Forecast future values from a univariate time series using Moirai-2.0 (small).

    Moirai-2.0 returns quantile forecasts rather than raw sample paths. To
    approximate forecast distributions, this function reconstructs an empirical
    inverse CDF from the model’s quantiles and generates Monte Carlo samples.

    Parameters
    series : pd.Series
        Univariate time series. The last `context_length` observations are used
        as conditioning context.
    context_length : int, optional
        Number of past observations to condition the forecast on 
        (e.g. 66 = 3 months of trading days). Default is 66.
    prediction_days : int, optional
        Number of steps to forecast into the future 
        (e.g. 22 = 1 month). Default is 22.
    n_samples : int, optional
        Number of forecast paths to simulate via inverse-CDF sampling. 
        Default is 100.

    Returns
    low : np.ndarray
        10th percentile forecast across horizons (lower bound of 80% CI).
    median : np.ndarray
        50th percentile forecast across horizons (median path).
    high : np.ndarray
        90th percentile forecast across horizons (upper bound of 80% CI).
    samples : np.ndarray
        Simulated forecast sample paths of shape (n_samples, prediction_days),
        generated from interpolated quantiles.
    base_price : float
        The last observed value in the conditioning window.
    """
    if len(series) < context_length:
        raise ValueError(f"series length {len(series)} < context_length {context_length}")

    # conditioning window
    ctx = series[-context_length:].astype(float)

    # build a simple daily GluonTS dataset
    df = pd.DataFrame(
        {
            "Date": pd.date_range(start="2000-01-01", periods=context_length, freq="D"),
            "ID": 0,
            "Value": ctx.values,
        }
    )
    dataset = PandasDataset.from_long_dataframe(
        df, target="Value", item_id="ID", timestamp="Date", freq="D"
    )

    # Moirai 2.0 forecaster (no num_samples here)
    model = Moirai2Forecast(
        module=Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small"),
        prediction_length=prediction_days,
        context_length=context_length,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )

    predictor = model.create_predictor(batch_size=1)

    # This returns a QuantileForecast (not SampleForecast)
    fc = next(predictor.predict(dataset))  # any num_samples passed here is ignored internally

    # Quantile grid to reconstruct an empirical inverse CDF
    q_levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=float)

    # Collect quantile curves for each horizon t
    # fc.quantile(p) -> shape (prediction_length,)
    q_matrix = np.stack([np.asarray(fc.quantile(float(p))).reshape(-1) for p in q_levels], axis=1)
    # shape: (prediction_days, len(q_levels))

    # enforce monotonicity across quantiles per horizon (guard against tiny numeric violations)
    q_matrix = np.maximum.accumulate(q_matrix, axis=1)

    # Inverse-CDF sampling per horizon via linear interpolation
    samples = np.empty((n_samples, prediction_days), dtype=float)
    rng = np.random.default_rng()
    for t in range(prediction_days):
        inv_cdf = interp1d(
            q_levels, q_matrix[t, :],
            kind="linear", fill_value="extrapolate", bounds_error=False,
            assume_sorted=True
        )
        u = rng.random(n_samples)
        samples[:, t] = inv_cdf(u)

    # Use model's quantiles for low/median/high (consistent with provider)
    low    = q_matrix[:, 0]      # 0.1
    median = q_matrix[:, 4]      # 0.5
    high   = q_matrix[:, -1]     # 0.9

    base_price = float(ctx.iloc[-1])
    return low, median, high, samples, base_price
