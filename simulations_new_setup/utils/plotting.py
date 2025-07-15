# ------------------------------------------------------------------------------
# Innosuisse Project: Usability of Transformer Models for Modelling Commodity Markets
# Date: June 29, 2025
# Authors: Peter Gruber (peter.gruber@usi.ch), Alessandro Dodon (alessandro.dodon@usi.ch)
#
# This script defines the plotting and formatting functions used in all experiments.
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Packages
# ------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ------------------------------------------------------------------------------
# Forecast Plot (with sequential integer index)
# ------------------------------------------------------------------------------
def plot_forecast(series, low, median, high, figsize=(10, 4),
                  dgp_type=None, context_length=None, path=""):
    """
    Plot forecast results with 80% CI and median line.

    Parameters:
    - series: pd.Series, original simulated series
    - low, median, high: forecast percentiles
    - figsize: tuple, optional
    - dgp_type: optional string
    - context_length: optional int
    - path: string, optional save path (e.g., 'plots/forecast.png')
    """
    context_end = len(series) - 1
    forecast_index = np.arange(context_end + 1, context_end + 1 + len(median))

    plt.figure(figsize=figsize)
    plt.plot(series.index, series.values, label="Simulated Series", color='black')
    plt.plot([context_end, forecast_index[0]], [series.iloc[-1], median[0]], color='gray', linestyle='--', alpha=0.6)
    plt.plot(forecast_index, median, linestyle='--', color='gray', alpha=0.6, label="Median Forecast")
    plt.fill_between(forecast_index, low, high, color='gray', alpha=0.1, label="80% CI")
    plt.plot(context_end, series.iloc[-1], 'o', color='black', label="Forecast Start")

    title = "Forecast"
    if dgp_type and context_length is not None:
        title += f"\nDGP: {dgp_type}, Context Length: {context_length}"
    plt.title(title)

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()

    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight')


# ------------------------------------------------------------------------------
# KDEs on Multiple Days
# ------------------------------------------------------------------------------
def plot_daily_return_kdes(samples, selected_days,
                           dgp_type=None, context_length=None, path=""):
    """
    Plot KDEs of forecasted daily returns for selected forecast days.

    Parameters:
    - samples: np.ndarray of shape (n_samples, n_days)
    - selected_days: list of int
    - dgp_type: optional string
    - context_length: optional int
    - path: string, optional save path (e.g., 'plots/kde_multiple.png')
    """
    daily_returns = samples[:, 1:] / samples[:, :-1] - 1

    plt.figure(figsize=(15, 4))
    for i, day_index in enumerate(selected_days):
        returns = daily_returns[:, day_index]
        lower, upper = np.percentile(returns, [0.05, 99.95])

        plt.subplot(1, len(selected_days), i + 1)
        sns.kdeplot(returns, color='black', linewidth=2, bw_adjust=2)
        plt.xlim(lower, upper)
        plt.xlabel("Return", color='black')
        plt.ylabel("Density", color='black')
        plt.title(f"KDE - Day {day_index + 2}", color='black')
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))

    if dgp_type and context_length is not None:
        plt.suptitle(f"KDEs of Returns\nDGP: {dgp_type}, Context Length: {context_length}", y=1.05)

    plt.tight_layout()

    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight')


# ------------------------------------------------------------------------------
# KDE Single Day
# ------------------------------------------------------------------------------
def plot_daily_return_kde(samples, day_index,
                          dgp_type=None, context_length=None, path=""):
    """
    Plot kernel density estimate of the daily return from Day t to Day t+1.

    Parameters:
    - samples: np.ndarray of shape (n_samples, n_days)
    - day_index: int, index of the daily return (0-based)
    - dgp_type: optional string
    - context_length: optional int
    - path: string, optional save path
    """
    daily_returns = samples[:, 1:] / samples[:, :-1] - 1
    returns = daily_returns[:, day_index]
    lower, upper = np.percentile(returns, [0.05, 99.95])

    plt.figure(figsize=(8, 5))
    sns.kdeplot(returns, color='black', linewidth=2, bw_adjust=2)
    plt.xlim(lower, upper)

    title = f"KDE - Day {day_index + 2}"
    if dgp_type and context_length is not None:
        title += f"\nDGP: {dgp_type}, Context Length: {context_length}"
    plt.title(title, color='black')

    plt.xlabel("Return", color='black')
    plt.ylabel("Density", color='black')
    plt.tight_layout()

    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight')


# ------------------------------------------------------------------------------
# CDFs on Multiple Days
# ------------------------------------------------------------------------------
def plot_daily_return_cdfs(samples, selected_days,
                           dgp_type=None, context_length=None, path=""):
    """
    Plot empirical CDFs of forecasted daily returns for selected forecast days.

    Parameters:
    - samples: np.ndarray of shape (n_samples, n_days)
    - selected_days: list of int
    - dgp_type: optional string
    - context_length: optional int
    - path: string, optional save path (e.g., 'plots/cdf_multiple.png')
    """
    daily_returns = samples[:, 1:] / samples[:, :-1] - 1

    plt.figure(figsize=(15, 4))
    for i, day_index in enumerate(selected_days):
        returns = daily_returns[:, day_index]
        bin_edges = np.histogram_bin_edges(returns, bins=30)

        plt.subplot(1, len(selected_days), i + 1)
        plt.hist(
            returns,
            bins=bin_edges,
            cumulative=True,
            density=True,
            histtype='step',
            color='black'
        )
        plt.xlim(bin_edges[0], bin_edges[-1])
        plt.xlabel("Return", color='black')
        plt.ylabel("Cumulative Probability", color='black')
        plt.title(f"CDF - Day {day_index + 2}", color='black')

    if dgp_type and context_length is not None:
        plt.suptitle(f"CDFs of Returns\nDGP: {dgp_type}, Context Length: {context_length}", y=1.05)

    plt.tight_layout()

    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight')


# ------------------------------------------------------------------------------
# CDF Single Day
# ------------------------------------------------------------------------------
def plot_day_return_cdf(samples, day_index,
                        dgp_type=None, context_length=None, path=""):
    """
    Plot the empirical cumulative distribution function (CDF) for the daily return.

    Parameters:
    - samples: np.ndarray of shape (n_samples, n_days)
    - day_index: int, index of the daily return (0-based)
    - dgp_type: optional string
    - context_length: optional int
    - path: string, optional save path
    """
    daily_returns = samples[:, 1:] / samples[:, :-1] - 1
    returns = daily_returns[:, day_index]

    num_bins = 30
    bin_edges = np.histogram_bin_edges(returns, bins=num_bins)
    
    plt.figure(figsize=(7, 5))
    plt.hist(
        returns,
        bins=bin_edges,
        cumulative=True,
        density=True,
        histtype='step',
        color='black'
    )

    title = f"Empirical CDF - Day {day_index + 2}"
    if dgp_type and context_length is not None:
        title += f"\nDGP: {dgp_type}, Context Length: {context_length}"
    plt.title(title, color='black')

    plt.xlabel("Return", color='black')
    plt.ylabel("Cumulative Probability", color='black')
    plt.xlim(bin_edges[0], bin_edges[-1])
    plt.tight_layout()

    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight')


# ------------------------------------------------------------------------------
# Plotting Divergence Test for Multiple Days
# ------------------------------------------------------------------------------
def plot_multiple_return_kde_comparison(dgp_samples, model_samples, selected_days,
                                        dgp_type=None, context_length=None, path=""):
    """
    Plot KDE comparisons of return distributions for multiple forecast days.

    Parameters:
    - dgp_samples: np.ndarray (n_samples, n_days)
    - model_samples: np.ndarray (n_samples, n_days)
    - selected_days: list of int
    - dgp_type: optional string
    - context_length: optional int
    - path: string, optional save path
    """
    dgp_returns = dgp_samples[:, 1:] / dgp_samples[:, :-1] - 1
    model_returns = model_samples[:, 1:] / model_samples[:, :-1] - 1

    plt.figure(figsize=(15, 4))

    for i, day_index in enumerate(selected_days):
        plt.subplot(1, len(selected_days), i + 1)

        dgp_day = dgp_returns[:, day_index]
        model_day = model_returns[:, day_index]

        all_values = np.concatenate([dgp_day, model_day])
        lower, upper = np.percentile(all_values, [0.05, 99.95])

        sns.kdeplot(dgp_day, label="DGP", fill=True, color="green", alpha=0.3, linewidth=2, bw_adjust=2)
        sns.kdeplot(model_day, label="Model", fill=True, color="blue", alpha=0.3, linewidth=2, bw_adjust=2)

        plt.xlim(lower, upper)
        plt.xlabel("Return")
        plt.ylabel("Density")
        plt.title(f"Day {day_index + 2}")
        plt.legend(loc="upper right")

    if dgp_type and context_length is not None:
        plt.suptitle(f"Density Comparison\nDGP: {dgp_type}, Context Length: {context_length}", y=1.05)

    plt.tight_layout()

    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight')


# ------------------------------------------------------------------------------
# Plotting Divergence Test Single Day
# ------------------------------------------------------------------------------
def plot_return_kde_comparison(dgp_samples, model_samples, day_index=10,
                               dgp_type=None, context_length=None, path=""):
    """
    Compare KDEs of return distributions for one forecast day between DGP and Model.

    Parameters:
    - dgp_samples: np.ndarray (n_samples, n_days)
    - model_samples: np.ndarray (n_samples, n_days)
    - day_index: int, index of the forecast day (0-based)
    - dgp_type: optional string label for the DGP
    - context_length: optional int label
    - path: string, optional path to save the figure (e.g. "plots/kde_day10.png")
    """
    dgp_returns = dgp_samples[:, 1:] / dgp_samples[:, :-1] - 1
    model_returns = model_samples[:, 1:] / model_samples[:, :-1] - 1

    dgp_day = dgp_returns[:, day_index]
    model_day = model_returns[:, day_index]

    all_values = np.concatenate([dgp_day, model_day])
    lower, upper = np.percentile(all_values, [0.05, 99.95])

    plt.figure(figsize=(8, 5))
    sns.kdeplot(dgp_day, label="DGP", fill=True, color="green", alpha=0.3, linewidth=2, bw_adjust=2)
    sns.kdeplot(model_day, label="Model", fill=True, color="blue", alpha=0.3, linewidth=2, bw_adjust=2)
    plt.xlim(lower, upper)

    title = f"Return Density: Day {day_index + 2}"
    if dgp_type and context_length is not None:
        title += f"\nDGP: {dgp_type}, Context Length: {context_length}"

    plt.title(title)
    plt.xlabel("Daily Return")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight')

