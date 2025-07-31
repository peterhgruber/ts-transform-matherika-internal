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
                  dgp_type=None, context_length=None, path="",
                  is_price_data=True):
    """
    Plot forecast with 80% CI and median line. Supports price or return mode.
    """
    plt.figure(figsize=figsize)

    if is_price_data:
        context_end = len(series) - 1
        forecast_index = np.arange(context_end + 1, context_end + 1 + len(median))

        plt.plot(series.index, series.values, label="Simulated Series", color='black')
        plt.plot([context_end, forecast_index[0]], [series.iloc[-1], median[0]], color='gray', linestyle='--', alpha=0.6)
        plt.plot(forecast_index, median, linestyle='--', color='gray', alpha=0.6, label="Median Forecast")
        plt.fill_between(forecast_index, low, high, color='gray', alpha=0.1, label="80% CI")
        plt.plot(context_end, series.iloc[-1], 'o', color='black', label="Forecast Start")
        plt.xlabel("Time")
        plt.ylabel("Price")

    else:
        series_length = len(series) - 1
        full_returns = series.values
        forecast_index = np.arange(series_length + 1, series_length + 1 + len(median))

        # Plot the true returns before forecast
        plt.plot(np.arange(series_length + 1), full_returns, label="Simulated Returns", color='black')

        # Connect to forecast
        plt.plot([series_length, forecast_index[0]], [full_returns[-1], median[0]], color='gray', linestyle='--', alpha=0.6)

        plt.plot(forecast_index, median, linestyle='--', color='gray', alpha=0.6, label="Median Forecast")
        plt.fill_between(forecast_index, low, high, color='gray', alpha=0.1, label="80% CI")
        plt.plot(series_length, full_returns[-1], 'o', color='black', label="Forecast Start")

        plt.xlabel("Time")
        plt.ylabel("Return")

    title = "Forecast"
    if dgp_type and context_length is not None:
        title += f"\nDGP: {dgp_type}, Context Length: {context_length}"
    plt.title(title)

    plt.legend()
    plt.tight_layout()

    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()


# ------------------------------------------------------------------------------
# KDEs on Multiple Days
# ------------------------------------------------------------------------------
def plot_daily_return_kdes(samples, selected_days,
                           dgp_type=None, context_length=None, path="",
                           is_price_data=True):
    """
    Plot KDEs of forecasted daily returns for selected forecast days.
    """
    daily_returns = samples[:, 1:] / samples[:, :-1] - 1 if is_price_data else samples

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
        plt.close()


# ------------------------------------------------------------------------------
# CDFs on Multiple Days
# ------------------------------------------------------------------------------
def plot_daily_return_cdfs(samples, selected_days,
                           dgp_type=None, context_length=None, path="",
                           is_price_data=True):
    """
    Plot empirical CDFs of forecasted daily returns for selected forecast days.
    """
    daily_returns = samples[:, 1:] / samples[:, :-1] - 1 if is_price_data else samples

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
        plt.close()


# ------------------------------------------------------------------------------
# Plotting Divergence Test 
# ------------------------------------------------------------------------------
def plot_multiple_return_kde_comparison(dgp_samples, model_samples, selected_days,
                                        dgp_type=None, context_length=None, path="",
                                        is_price_data=True):
    """
    Plot KDE comparisons of return distributions for multiple forecast days.
    """
    dgp_returns = dgp_samples[:, 1:] / dgp_samples[:, :-1] - 1 if is_price_data else dgp_samples
    model_returns = model_samples[:, 1:] / model_samples[:, :-1] - 1 if is_price_data else model_samples

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
        plt.close()


# ------------------------------------------------------------------------------
# KL Divergence vs Context Length Plot 
# ------------------------------------------------------------------------------
def plot_kl_vs_context(kl_dataframe, output_path, target_type_label):
    """
    Create bar plots of KL divergence vs context length for each model and day.

    Parameters:
        kl_dataframe (pd.DataFrame): Must include:
            ["context_length", "dgp_type", "model_name", "day", "kl_divergence"]
        output_path (Path): Folder where to save plots
        target_type_label (str): Either "prices" or "returns"
    """
    output_path.mkdir(parents=True, exist_ok=True)
    grouped_models = kl_dataframe.groupby("model_name")

    for model_name, model_df in grouped_models:
        grouped_days = model_df.groupby("day")

        for day_label, day_df in grouped_days:
            plt.figure(figsize=(10, 5))

            sns.barplot(
                data=day_df,
                x="context_length",
                y="kl_divergence",
                hue="dgp_type",
                ci=None,
                palette="tab10"
            )

            plt.title(f"{model_name} — {target_type_label} — {day_label}")
            plt.xlabel("Context Length")
            plt.ylabel("KL Divergence")
            plt.legend(title="DGP", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()

            filename = output_path / f"kl_barplot_{model_name}_{target_type_label}_{day_label.replace(' ', '')}.png"
            plt.savefig(filename, dpi=300)
            plt.close()


# ------------------------------------------------------------------------------
# KL Divergence vs Models 
# ------------------------------------------------------------------------------
def plot_model_comparison_bar_avg(df_kl, output_path, target_type_label, selected_days):
    """
    Plot average KL divergence per model (horizontal bars), grouped by DGP and context length.
    Models are shown via legend (cleaner than x-axis).
    """
    output_path.mkdir(parents=True, exist_ok=True)

    df_avg = (
        df_kl[df_kl["day"].isin([f"Day {i + 2}" for i in selected_days])]
        .groupby(["context_length", "dgp_type", "model_name"])["kl_divergence"]
        .mean()
        .reset_index(name="avg_kl")
    )

    for (dgp_name, context_val), group_df in df_avg.groupby(["dgp_type", "context_length"]):
        plot_df = group_df.sort_values("avg_kl")

        plt.figure(figsize=(8, 6))
        bars = plt.bar(
            x=np.arange(len(plot_df)),
            height=plot_df["avg_kl"],
            color=sns.color_palette("viridis", len(plot_df))
        )

        # Legend with model names
        plt.legend(
            bars,
            plot_df["model_name"],
            title="Model",
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.
        )

        plt.ylabel("Average KL Divergence")
        plt.xlabel("Model Index")
        plt.title(f"{dgp_name} — {target_type_label} — Context {context_val}")
        plt.tight_layout()

        filename = output_path / f"model_bar_avg_{target_type_label}_{dgp_name}_context{context_val}.png"
        plt.savefig(filename, dpi=300)
        plt.close()


# ------------------------------------------------------------------------------
# KL Divergence vs Volatility  
# ------------------------------------------------------------------------------
def plot_kl_bar_by_vol(df_kl, output_path, target_type_label, fixed_context):
    df_filtered = df_kl[df_kl["context_length"] == fixed_context]
    grouped = df_filtered.groupby("model_name")

    for model_name, model_df in grouped:
        for day, day_df in model_df.groupby("day"):
            df_day = day_df.copy()
            df_day = df_day.sort_values("volatility")

            plt.figure(figsize=(8, 5))
            sns.barplot(
                data=df_day,
                x="volatility",
                y="kl_divergence",
                hue="dgp_type",
                palette="viridis"
            )
            plt.title(f"{model_name} — {target_type_label} — {day} (context {fixed_context})")
            plt.xlabel("Annualized Volatility")
            plt.ylabel("KL Divergence")
            plt.legend(title="DGP Type", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()

            filename = output_path / f"kl_bar_vol_{model_name}_{target_type_label}_{day.replace(' ', '')}_context{fixed_context}.png"
            plt.savefig(filename, dpi=300)
            plt.close()

