"""!
@file   plotting.py
@brief  This file contains various plotting functions to visualize diagnostics and
comparisons for statistical distributions and sampling algorithms.

The functions are tailored for analyzing Markov chain samples and their convergence behaviour using trace plots,
autocorrelation plots, corner plots, etc.

@author Ivo Petrov
@date   13/03/2024
"""
import matplotlib.pyplot as plt
import corner
import scipy
import arviz as az
import numpy as np


def trace_plot(chain, param_names, title=None, out_path=None, max_len=5000):
    """! Displays the trace plot for a generated Markov chain.

    @param chain        The generated Markov Chain
    @param param_names  The names of the parameters.
    @param title        Customizable plot title.
    @param out_path     The output path of where to save the plot.
    @param max_len"""
    plt.figure(figsize=(8, 3))

    # Plot a trace for each parameter
    for i, param in enumerate(param_names):
        # Plot normalized traces (shift for visibility)
        plt.plot(
            chain[:max_len, i] / chain[:max_len, i].max() + i, label=param, alpha=0.35
        )
        plt.axhline(
            i + chain[:max_len, i].mean() / chain[:max_len, i].max(),
            color="gray",
            linestyle="--",
            alpha=0.7,
        )

    plt.xlabel("Chain index")
    plt.ylabel("Parameter Value (max-scaled)")
    plt.suptitle(title)
    plt.legend()
    plt.tight_layout()

    # Save or show plot
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path, dpi=300)


def autocorr_plot(chain, param_names, max_lag=1000, title=None, out_path=None):
    """! Displays the autocorrelation plots for a generated Markov chain.

    @param chain        The generated Markov Chain
    @param param_names  The names of the parameters.
    @param max_lag      The maximum lag being inspected.
    @param title        Customizable plot title.
    @param out_path     The output path of where to save the plot."""

    plt.rcParams.update({"font.size": 22})

    fig, ax = plt.subplots(1, len(param_names), figsize=(16, 6), dpi=120)
    # Create an Arviz dataset for compatibility
    dataset = az.from_dict(
        {param: chain[np.newaxis, :, i] for i, param in enumerate(param_names)}
    )
    # Plot the autocorrelation dependencies
    az.plot_autocorr(dataset, max_lag=max_lag, ax=ax)

    # Add labels and title
    ax[0].set_ylabel("Autocorrelation Coefficient")
    ax[0].set_ylim(-0.1, 1.0)
    fig.text(0.5, 0.04, "Series Lag", ha="center", va="center")
    plt.suptitle(title)
    plt.tight_layout()

    plt.rcParams.update({"font.size": 10})

    # Save or show plot
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path, dpi=300)


def corner_plot(samples, param_names, title=None, out_path=None):
    """! Displays the 2-dimensional and 1-dimensional plot

    @param chain        The isolated IID samples.
    @param param_names  The names of the parameters.
    @param title        Customizable plot title.
    @param out_path     The output path of where to save the plot."""
    # Corner plot with the quantiles corresponding to the median an 1 std.
    corner.corner(
        samples, labels=param_names, quantiles=[0.16, 0.5, 0.84], show_titles=True
    )
    plt.suptitle(title)
    plt.tight_layout()

    # Save or show plot
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path, dpi=300)


def plot_cauchy_convergence(out_path=None):
    """! A plot to compare the convergence of the Cauchy and Normal
    distribution means.

    @param out_path     The output path of where to save the plot."""

    np.random.seed(16)

    # Sample sizes to consider
    sample_sizes = np.exp(np.linspace(3, 10, 100)).astype(np.int32)
    cauchy_means = []
    normal_means = []

    for size in sample_sizes:
        # Generate samples for each sample size
        cauchy_samples = scipy.stats.cauchy.rvs(size=size)
        normal_samples = scipy.stats.norm.rvs(size=size)

        # Collect their means
        cauchy_means.append(np.mean(cauchy_samples))
        normal_means.append(np.mean(normal_samples))

    plt.rcParams.update({"font.size": 16})
    plt.figure(figsize=(10, 6))

    plt.plot(
        sample_sizes, cauchy_means, label="Cauchy Distribution Means", color="tab:red"
    )
    plt.plot(
        sample_sizes, normal_means, label="Normal Distribution Means", color="tab:blue"
    )

    plt.xlabel("Sample Size")
    plt.ylabel("Sample Mean")
    plt.title("Sample Mean Convergence of Cauchy vs Normal Distribution")

    # Rescale axes for clarity
    plt.yscale("symlog")
    plt.xscale("log")

    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.rcParams.update({"font.size": 10})

    # Save or show plot
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path, dpi=300)


def compare_prior(
    prior_pdfs, samples, param_names, x_ranges, y_scales, bins=30, out_path=None
):
    """! Compares multiple parameters' prior distributions with samples drawn for them.

    @param prior_pdfs   The priors of each parameter given as a function.
    @param samples      The drawn samples for each parameter.
    @param param_names  The names of the parameters.
    @param x_ranges     The ranges of the parameters for the sake of the plot.
    @param y_scales     How the y-axis should be scaled.
    @param bins         The number of bins for each histogram.
    @param out_path     The output path of where to save the plot."""

    num_params = len(prior_pdfs)
    plt.figure(figsize=(12, 4 * num_params))

    for i in range(num_params):
        ax = plt.subplot(num_params, 1, i + 1)

        x_values = np.linspace(x_ranges[i][0], x_ranges[i][1], 1000)
        pdf_values = prior_pdfs[i].pdf(x_values)

        # Plot the prior distribution PDF
        ax.plot(x_values, pdf_values, label=f"{param_names[i]} Prior PDF", lw=2)

        # Plot the histogram of the samples
        ax.hist(
            samples[i],
            bins=bins,
            density=True,
            alpha=0.5,
            label=f"{param_names[i]} Sample Histogram",
        )

        # Plot KDE of the samples
        kde = scipy.stats.gaussian_kde(samples[i])
        kde_values = kde(x_values)
        kde_values = np.where(kde_values < 1e-6, 1e-6, kde_values)
        ax.plot(
            x_values,
            kde_values,
            label=f"{param_names[i]} Sample KDE",
            color="tab:red",
        )

        ax.set_title(f"Prior and sample comparison for {param_names[i]}")
        ax.set_yscale(y_scales[i])
        ax.set_xlabel(param_names[i])
        ax.set_ylabel("Density")
        ax.legend()

    plt.tight_layout()

    # Save or show plot
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path, dpi=300)
