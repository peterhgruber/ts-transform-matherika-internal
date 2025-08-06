# ------------------------------------------------------------------------------
# Packages
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.stats import t
import warnings
warnings.filterwarnings('ignore')


# PART 1: SIMULATING PRICES


# ------------------------------------------------------------------------------
# Simulate price time series with seasonal component
# ------------------------------------------------------------------------------
def simulate_seasonal_prices(time_steps=100, initial_value=100.0,
                             amplitude=0.05, frequency=1/20,
                             trend=0.0002, noise_std=0.0, seed=None):
    """
    Simulates a price time series with a seasonal (cyclical) return component.

    Parameters:
    - time_steps: int, number of time steps (e.g., days)
    - initial_value: float, starting price
    - amplitude: float, amplitude of the sine wave in return space
    - frequency: float, cycles per time step (e.g., 1/20 = 1 cycle per 20 steps)
    - trend: float, optional constant trend in log returns
    - noise_std: float, standard deviation of Gaussian noise in return
    - seed: int or None

    Returns:
    - pd.Series of simulated prices
    """
    if seed is not None:
        np.random.seed(seed)

    time_index = np.arange(time_steps)
    seasonal_component = amplitude * np.sin(2 * np.pi * frequency * time_index)
    noise_component = np.random.normal(0.0, noise_std, size=time_steps)
    log_returns = trend + seasonal_component + noise_component
    log_prices = np.log(initial_value) + np.cumsum(log_returns)
    prices = np.exp(log_prices)

    return pd.Series(prices)


# ------------------------------------------------------------------------------
# Simulate constant price time series (flat line)
# ------------------------------------------------------------------------------
def simulate_constant_prices(time_steps=100, initial_value=100.0):
    """
    Simulates a constant price time series (no returns, no noise).

    Parameters:
    - time_steps: Number of time steps to simulate (e.g., trading days)
    - initial_value: Constant price level throughout

    Returns:
    - pd.Series of constant prices
    """
    prices = np.full(time_steps, initial_value)
    return pd.Series(prices)


# ------------------------------------------------------------------------------
# Simulate linear price time series with constant daily return
# ------------------------------------------------------------------------------
def simulate_linear_prices(time_steps=100, initial_value=100.0, daily_return=0.0005):
    """
    Simulates a price time series with a deterministic linear trend in log space.

    Parameters:
    - time_steps: Number of time steps to simulate (e.g., trading days)
    - initial_value: Starting price
    - daily_return: Constant log return per step (e.g., 0.0005 ≈ 0.05% daily)

    Returns:
    - pd.Series of simulated prices
    """
    log_prices = np.log(initial_value) + np.arange(time_steps) * daily_return
    prices = np.exp(log_prices)
    return pd.Series(prices)


# ------------------------------------------------------------------------------
# Simulate price time series using a T-GARCH(1,1) model for volatility
# ------------------------------------------------------------------------------
def simulate_t_garch_prices(time_steps=100, initial_value=100.0,
                            omega=0.01, alpha=0.1, beta=0.85,
                            volatility_start=0.02, degrees_freedom=10, seed=None):
    """
    Simulates a price time series using a T-GARCH(1,1) model with Student-t innovations.

    Parameters:
    - time_steps: Number of time steps to simulate (e.g., trading days)
    - initial_value: Initial price level
    - omega, alpha, beta: T-GARCH(1,1) parameters (per time step)
    - volatility_start: Starting daily volatility
    - degrees_freedom: Degrees of freedom for the Student-t distribution
    - seed: Optional seed for reproducibility

    Returns:
    - pd.Series of simulated prices
    """
    if seed is not None:
        np.random.seed(seed)

    prices = np.empty(time_steps)
    returns = np.empty(time_steps)
    volatility = np.empty(time_steps)

    prices[0] = initial_value
    volatility[0] = volatility_start
    returns[0] = 0.0

    for t_index in range(1, time_steps):
        shock = t.rvs(df=degrees_freedom)
        returns[t_index] = volatility[t_index - 1] * shock / np.sqrt(degrees_freedom / (degrees_freedom - 2))
        volatility[t_index] = np.sqrt(omega + alpha * returns[t_index - 1]**2 + beta * volatility[t_index - 1]**2)
        prices[t_index] = prices[t_index - 1] * np.exp(returns[t_index])

    return pd.Series(prices)


# ------------------------------------------------------------------------------
# Simulate price time series using a mixture of normal distributions
# ------------------------------------------------------------------------------
def simulate_mixture_normal_prices(time_steps=100, initial_value=100.0,
                                   means=[0.0, 0.0],
                                   std_devs=[0.01, 0.05],
                                   weights=[0.9, 0.1],
                                   seed=None):
    """
    Simulates a price time series using a mixture of normal distributions for returns.

    Parameters:
    - time_steps: Number of time steps to simulate (e.g., trading days)
    - initial_value: Initial price level
    - means: List of daily return means for each component
    - std_devs: List of daily standard deviations for each component
    - weights: List of probabilities for selecting each component
    - seed: Optional seed for reproducibility

    Returns:
    - pd.Series of simulated prices
    """
    if seed is not None:
        np.random.seed(seed)

    num_components = len(means)
    component_indices = np.random.choice(num_components, size=time_steps, p=weights)
    returns = np.array([
        np.random.normal(loc=means[i], scale=std_devs[i])
        for i in component_indices
    ])

    prices = np.empty(time_steps)
    prices[0] = initial_value

    for t_index in range(1, time_steps):
        prices[t_index] = prices[t_index - 1] * np.exp(returns[t_index])

    return pd.Series(prices)


# ------------------------------------------------------------------------------
# Simulate price time series using Geometric Brownian Motion
# ------------------------------------------------------------------------------
def simulate_gbm_prices(time_steps=100, initial_value=100.0,
                               drift=0.05, volatility=0.2, seed=None):
    """
    Simulates a price time series using Geometric Brownian Motion (GBM).

    Parameters:
    - time_steps: Number of time steps to simulate (e.g., trading days)
    - initial_value: Initial price level
    - drift: Per-step drift (e.g., daily drift)
    - volatility: Per-step volatility (e.g., daily volatility)
    - seed: Optional seed for reproducibility

    Returns:
    - pd.Series of simulated prices
    """
    if seed is not None:
        np.random.seed(seed)

    dt = 1
    shocks = np.random.normal(0, 1, size=time_steps)
    returns = (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * shocks
    log_prices = np.log(initial_value) + np.cumsum(returns)
    prices = np.exp(log_prices)

    return pd.Series(prices)


# PART 2: SIMULATING MULTIPLE PATHS

# ------------------------------------------------------------------------------
# Simulate multiple paths using seasonal series with noise
# ------------------------------------------------------------------------------
def forecast_seasonal_prices_paths(base_price, forecast_days=22, n_samples=1000,
                            amplitude=0.05, frequency=1/20,
                            trend=0.0002, noise_std=0.01, seed=None):
    """
    Forecast seasonal price paths from a given base price using a sine-based return model.

    Parameters:
    - base_price: float, starting price
    - forecast_days: int, forecast horizon
    - n_samples: int, number of paths
    - amplitude: float, sine wave amplitude (return space)
    - frequency: float, cycles per step
    - trend: float, constant return drift
    - noise_std: float, standard deviation of noise
    - seed: int or None

    Returns:
    - np.ndarray of shape (n_samples, forecast_days)
    """
    if seed is not None:
        np.random.seed(seed)

    time_index = np.arange(forecast_days)
    seasonal_component = amplitude * np.sin(2 * np.pi * frequency * time_index)

    samples = np.empty((n_samples, forecast_days))

    for i in range(n_samples):
        noise = np.random.normal(0.0, noise_std, size=forecast_days)
        log_returns = trend + seasonal_component + noise
        log_prices = np.log(base_price) + np.cumsum(log_returns)
        samples[i] = np.exp(log_prices)

    return samples


# ------------------------------------------------------------------------------
# Simulate multiple paths using a T-GARCH(1,1) model
# ------------------------------------------------------------------------------
def forecast_t_garch_prices_paths(base_price, forecast_days=22, n_samples=1000,
                           omega=0.01, alpha=0.1, beta=0.85,
                           last_volatility=0.02, last_return=0.0,
                           degrees_freedom=10, seed=None):
    """
    Forecast T-GARCH(1,1) paths from a starting price, return, and volatility.

    Parameters:
    - base_price: float, last observed price
    - forecast_days: int, forecast horizon
    - n_samples: int, number of simulated paths
    - omega, alpha, beta: GARCH parameters
    - last_volatility: float, volatility at forecast start
    - last_return: float, return at forecast start
    - degrees_freedom: degrees of freedom of t-distribution
    - seed: int or None

    Returns:
    - samples: np.ndarray of shape (n_samples, forecast_days)
    """
    if seed is not None:
        np.random.seed(seed)

    samples = np.empty((n_samples, forecast_days))

    for path_index in range(n_samples):
        prices = np.empty(forecast_days)
        volatility = np.empty(forecast_days)
        returns = np.empty(forecast_days)

        prices[0] = base_price
        volatility[0] = np.sqrt(omega + alpha * last_return**2 + beta * last_volatility**2)
        shock = t.rvs(df=degrees_freedom)
        returns[0] = volatility[0] * shock / np.sqrt(degrees_freedom / (degrees_freedom - 2))
        prices[0] = base_price * np.exp(returns[0])

        for t_index in range(1, forecast_days):
            shock = t.rvs(df=degrees_freedom)
            volatility[t_index] = np.sqrt(omega + alpha * returns[t_index - 1]**2 + beta * volatility[t_index - 1]**2)
            returns[t_index] = volatility[t_index] * shock / np.sqrt(degrees_freedom / (degrees_freedom - 2))
            prices[t_index] = prices[t_index - 1] * np.exp(returns[t_index])

        samples[path_index] = prices

    return samples


# ------------------------------------------------------------------------------
# Simulate multiple paths using a Mixture of Normal distributions
# ------------------------------------------------------------------------------
def forecast_mixture_normal_prices_paths(base_price, forecast_days=22, n_samples=1000,
                                   means=[0.0, -0.002],
                                   std_devs=[0.01, 0.03],
                                   weights=[0.90, 0.10],
                                   seed=None):
    """
    Forecast price paths using a mixture of normal distributions for returns.

    Parameters:
    - base_price: float, starting price for all paths
    - forecast_days: int, forecast horizon
    - n_samples: int, number of paths to simulate
    - means: list of means for each normal component
    - std_devs: list of std devs for each component
    - weights: list of probabilities for selecting each component
    - seed: int or None

    Returns:
    - samples: np.ndarray of shape (n_samples, forecast_days)
    """
    if seed is not None:
        np.random.seed(seed)

    num_components = len(means)
    total_steps = n_samples * forecast_days

    component_indices = np.random.choice(num_components, size=total_steps, p=weights)
    shocks = np.array([
        np.random.normal(loc=means[i], scale=std_devs[i])
        for i in component_indices
    ])
    returns = shocks.reshape(n_samples, forecast_days)

    log_prices = np.log(base_price) + np.cumsum(returns, axis=1)
    price_paths = np.exp(log_prices)

    return price_paths


# ------------------------------------------------------------------------------
# Simulate multiple paths using Geometric Brownian Motion
# ------------------------------------------------------------------------------
def forecast_gbm_prices_paths(base_price, forecast_days=22, n_samples=1000,
                       drift=0.0, volatility=0.2, seed=None):
    """
    Forecast GBM paths from a starting price.

    Parameters:
    - base_price: float, last observed price
    - forecast_days: int, forecast horizon
    - n_samples: int, number of simulated paths
    - drift: daily drift
    - volatility: daily volatility
    - seed: int or None

    Returns:
    - samples: np.ndarray of shape (n_samples, forecast_days)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = 1
    shocks = np.random.normal(0, 1, size=(n_samples, forecast_days))
    returns = (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * shocks
    log_prices = np.cumsum(returns, axis=1)
    log_prices += np.log(base_price)
    price_paths = np.exp(log_prices)

    return price_paths


# PART 2: SIMULATING RETURNS


# ------------------------------------------------------------------------------
# Simulate constant return series
# ------------------------------------------------------------------------------
def simulate_constant_returns(time_steps=100, daily_return=0.0):
    """
    Simulates a constant daily return time series.

    Parameters:
    - time_steps: int, number of time steps (e.g., days)
    - daily_return: float, constant return value per step

    Returns:
    - pd.Series of shape (time_steps,) representing constant daily returns
    """
    return pd.Series(np.full(time_steps, daily_return))


# ------------------------------------------------------------------------------
# Simulate linear return series (identical to constant, for compatibility)
# ------------------------------------------------------------------------------
def simulate_linear_returns(time_steps=100, daily_return=0.0005):
    """
    Simulates a deterministic linear return time series (log space).

    Parameters:
    - time_steps: int, number of time steps (e.g., days)
    - daily_return: float, constant log return per step

    Returns:
    - pd.Series of shape (time_steps,) representing constant returns
    """
    return pd.Series(np.full(time_steps, daily_return))


# ------------------------------------------------------------------------------
# Simulate GBM returns series
# ------------------------------------------------------------------------------
def simulate_gbm_returns(time_steps=100, drift=0.05, volatility=0.2, seed=None):
    """
    Simulates log returns from a Geometric Brownian Motion process.

    Parameters:
    - time_steps: int, number of time steps (e.g., days)
    - drift: float, expected daily return
    - volatility: float, daily return volatility
    - seed: int or None, random seed for reproducibility

    Returns:
    - pd.Series of shape (time_steps,) representing simulated returns
    """
    if seed is not None:
        np.random.seed(seed)

    dt = 1
    shocks = np.random.normal(0, 1, size=time_steps)
    returns = (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * shocks

    return pd.Series(returns)


# ------------------------------------------------------------------------------
# Simulate mixture normal returns series
# ------------------------------------------------------------------------------
def simulate_mixture_normal_returns(time_steps=100,
                                    means=[0.0, 0.0],
                                    std_devs=[0.01, 0.05],
                                    weights=[0.9, 0.1],
                                    seed=None):
    """
    Simulates a return series from a mixture of normal distributions.

    Parameters:
    - time_steps: int, number of time steps (e.g., days)
    - means: list of float, means of each mixture component
    - std_devs: list of float, standard deviations of each component
    - weights: list of float, selection probabilities for each component
    - seed: int or None, random seed

    Returns:
    - pd.Series of shape (time_steps,) representing simulated returns
    """
    if seed is not None:
        np.random.seed(seed)

    num_components = len(means)
    component_indices = np.random.choice(num_components, size=time_steps, p=weights)
    returns = np.array([
        np.random.normal(loc=means[i], scale=std_devs[i])
        for i in component_indices
    ])

    return pd.Series(returns)


# ------------------------------------------------------------------------------
# Simulate seasonal returns series
# ------------------------------------------------------------------------------
def simulate_seasonal_returns(time_steps=100,
                              amplitude=0.05, frequency=1/20,
                              trend=0.0002, noise_std=0.0, seed=None):
    """
    Simulates a return time series with a sinusoidal seasonal pattern.

    Parameters:
    - time_steps: int, number of time steps (e.g., days)
    - amplitude: float, amplitude of the sinusoidal component
    - frequency: float, frequency of the cycle (e.g., 1/20 = one cycle per 20 days)
    - trend: float, constant daily drift
    - noise_std: float, standard deviation of Gaussian noise
    - seed: int or None, random seed

    Returns:
    - pd.Series of shape (time_steps,) representing simulated returns
    """
    if seed is not None:
        np.random.seed(seed)

    time_index = np.arange(time_steps)
    seasonal_component = amplitude * np.sin(2 * np.pi * frequency * time_index)
    noise_component = np.random.normal(0.0, noise_std, size=time_steps)
    returns = trend + seasonal_component + noise_component

    return pd.Series(returns)


# ------------------------------------------------------------------------------
# Simulate T-GARCH returns series
# ------------------------------------------------------------------------------
def simulate_t_garch_returns(time_steps=100,
                             omega=0.01, alpha=0.1, beta=0.85,
                             volatility_start=0.02, degrees_freedom=10, seed=None):
    """
    Simulates returns from a T-GARCH(1,1) volatility process with Student-t innovations.

    Parameters:
    - time_steps: int, number of time steps (e.g., days)
    - omega, alpha, beta: float, GARCH model parameters
    - volatility_start: float, initial volatility
    - degrees_freedom: int, degrees of freedom for the t-distribution
    - seed: int or None, random seed

    Returns:
    - pd.Series of shape (time_steps,) representing simulated returns
    """
    if seed is not None:
        np.random.seed(seed)

    returns = np.empty(time_steps)
    volatility = np.empty(time_steps)

    volatility[0] = volatility_start
    returns[0] = 0.0

    for t_index in range(1, time_steps):
        shock = t.rvs(df=degrees_freedom)
        returns[t_index] = volatility[t_index - 1] * shock / np.sqrt(degrees_freedom / (degrees_freedom - 2))
        volatility[t_index] = np.sqrt(omega + alpha * returns[t_index - 1]**2 + beta * volatility[t_index - 1]**2)

    return pd.Series(returns)


# ------------------------------------------------------------------------------
# Forecast GBM return paths
# ------------------------------------------------------------------------------
def forecast_gbm_returns_paths(forecast_days=22, n_samples=1000,
                               drift=0.0, volatility=0.2, seed=None):
    """
    Forecasts return paths using Geometric Brownian Motion dynamics.

    Parameters:
    - forecast_days: int, forecast horizon
    - n_samples: int, number of forecasted paths
    - drift: float, expected daily return
    - volatility: float, daily return volatility
    - seed: int or None

    Returns:
    - np.ndarray of shape (n_samples, forecast_days)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = 1
    shocks = np.random.normal(0, 1, size=(n_samples, forecast_days))
    returns = (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * shocks

    return returns


# ------------------------------------------------------------------------------
# Forecast mixture normal return paths
# ------------------------------------------------------------------------------
def forecast_mixture_normal_returns_paths(forecast_days=22, n_samples=1000,
                                          means=[0.0, -0.002],
                                          std_devs=[0.01, 0.03],
                                          weights=[0.90, 0.10],
                                          seed=None):
    """
    Forecasts return paths from a mixture of normal distributions.

    Parameters:
    - forecast_days: int, forecast horizon
    - n_samples: int, number of paths
    - means: list of float, means of mixture components
    - std_devs: list of float, std devs of components
    - weights: list of float, probabilities of each component
    - seed: int or None

    Returns:
    - np.ndarray of shape (n_samples, forecast_days)
    """
    if seed is not None:
        np.random.seed(seed)

    num_components = len(means)
    total_steps = n_samples * forecast_days

    component_indices = np.random.choice(num_components, size=total_steps, p=weights)
    shocks = np.array([
        np.random.normal(loc=means[i], scale=std_devs[i])
        for i in component_indices
    ])
    returns = shocks.reshape(n_samples, forecast_days)

    return returns


# ------------------------------------------------------------------------------
# Forecast seasonal return paths
# ------------------------------------------------------------------------------
def forecast_seasonal_returns_paths(forecast_days=22, n_samples=1000,
                                    amplitude=0.05, frequency=1/20,
                                    trend=0.0002, noise_std=0.01, seed=None):
    """
    Forecasts return paths with a sinusoidal seasonal pattern plus noise.

    Parameters:
    - forecast_days: int, forecast horizon
    - n_samples: int, number of return paths
    - amplitude: float, amplitude of sine wave
    - frequency: float, cycles per step
    - trend: float, constant drift
    - noise_std: float, standard deviation of noise
    - seed: int or None

    Returns:
    - np.ndarray of shape (n_samples, forecast_days)
    """
    if seed is not None:
        np.random.seed(seed)

    time_index = np.arange(forecast_days)
    seasonal_component = amplitude * np.sin(2 * np.pi * frequency * time_index)

    samples = np.empty((n_samples, forecast_days))

    for i in range(n_samples):
        noise = np.random.normal(0.0, noise_std, size=forecast_days)
        samples[i] = trend + seasonal_component + noise

    return samples


# ------------------------------------------------------------------------------
# Forecast T-GARCH return paths
# ------------------------------------------------------------------------------
def forecast_t_garch_returns_paths(forecast_days=22, n_samples=1000,
                                   omega=0.01, alpha=0.1, beta=0.85,
                                   last_volatility=0.02, last_return=0.0,
                                   degrees_freedom=10, seed=None):
    """
    Forecasts return paths using a T-GARCH(1,1) process with t-distributed shocks.

    Parameters:
    - forecast_days: int, forecast horizon
    - n_samples: int, number of paths
    - omega, alpha, beta: float, GARCH model parameters
    - last_volatility: float, initial volatility
    - last_return: float, last observed return
    - degrees_freedom: int, degrees of freedom of t-distribution
    - seed: int or None

    Returns:
    - np.ndarray of shape (n_samples, forecast_days)
    """
    if seed is not None:
        np.random.seed(seed)

    samples = np.empty((n_samples, forecast_days))

    for path_index in range(n_samples):
        returns = np.empty(forecast_days)
        volatility = np.empty(forecast_days)

        volatility[0] = np.sqrt(omega + alpha * last_return**2 + beta * last_volatility**2)
        shock = t.rvs(df=degrees_freedom)
        returns[0] = volatility[0] * shock / np.sqrt(degrees_freedom / (degrees_freedom - 2))

        for t_index in range(1, forecast_days):
            shock = t.rvs(df=degrees_freedom)
            volatility[t_index] = np.sqrt(omega + alpha * returns[t_index - 1]**2 + beta * volatility[t_index - 1]**2)
            returns[t_index] = volatility[t_index] * shock / np.sqrt(degrees_freedom / (degrees_freedom - 2))

        samples[path_index] = returns

    return samples
