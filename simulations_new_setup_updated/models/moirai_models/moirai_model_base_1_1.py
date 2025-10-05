# ------------------------------------------------------------------------------
# Packages
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule


# ------------------------------------------------------------------------------
# Moirai-1.1 forecast function (base)
# ------------------------------------------------------------------------------
def forecast_moirai(series, context_length=66, prediction_days=22, n_samples=100):
    """
    Forecast future paths from a univariate time series using Moirai-1.1 (base).

    Parameters
    ----------
    series : pd.Series
        A univariate time series. The function will use the last `context_length` values for prediction.
    context_length : int
        Number of past observations to condition the forecast on.
    prediction_days : int
        Number of steps to forecast into the future.
    n_samples : int
        Number of forecast paths to simulate.

    Returns
    -------
    low : np.ndarray
        10th percentile across forecasted paths (lower bound of 80% confidence interval).
    median : np.ndarray
        50th percentile forecast (median forecast path).
    high : np.ndarray
        90th percentile (upper bound of 80% confidence interval).
    samples : np.ndarray
        Forecasted sample paths of shape (n_samples, prediction_days).
    base_price : float
        The last observed value in the context window.
    """
    # Extract the conditioning window from the input series
    context_series = series[-context_length:]

    # Format into a GluonTS-compatible PandasDataset
    df_input = pd.DataFrame({
        "Date": pd.date_range(start="2000-01-01", periods=context_length, freq="D"),
        "ID": 0,
        "Value": context_series.values
    })

    dataset = PandasDataset.from_long_dataframe(
        df_input,
        target="Value",
        item_id="ID",
        timestamp="Date",
        freq="D"
    )

    # Initialize Moirai-1.1 forecasting model (base)
    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-base"),
        prediction_length=prediction_days,
        context_length=context_length,
        patch_size="auto",
        num_samples=n_samples,
        target_dim=1,
        feat_dynamic_real_dim=0,         # No known future features
        past_feat_dynamic_real_dim=0     # No past dynamic features
    )

    # Make forecasts
    predictor = model.create_predictor(batch_size=1)
    forecast = next(predictor.predict(dataset))

    # Extract forecast samples and compute percentiles
    samples = forecast.samples  # shape: (n_samples, prediction_days)
    low, median, high = np.quantile(samples, [0.1, 0.5, 0.9], axis=0)

    # Base value for return computation (last observed value)
    base_price = context_series.iloc[-1]

    return low, median, high, samples, base_price
