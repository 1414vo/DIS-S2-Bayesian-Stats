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
    for i, param in enumerate(param_names):
        plt.plot(
            chain[:max_len, i] / chain[:max_len, i].max() + i, label=param, alpha=0.35
        )
        plt.axhline(i, color="gray", linestyle="--", alpha=0.7)

    plt.xlabel("Chain index")
    plt.ylabel("Parameter Value (max-scaled)")
    plt.suptitle(title)
    plt.legend()
    plt.tight_layout()
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
    dataset = az.from_dict(
        {param: chain[np.newaxis, :, i] for i, param in enumerate(param_names)}
    )
    az.plot_autocorr(dataset, max_lag=max_lag, ax=ax)

    # Add labels and title
    ax[0].set_ylabel("Autocorrelation Coefficient")
    ax[0].set_ylim(-0.1, 1.0)
    fig.text(0.5, 0.04, "Series Lag", ha="center", va="center")
    plt.suptitle(title)
    plt.tight_layout()

    plt.rcParams.update({"font.size": 10})
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
    corner.corner(
        samples, labels=param_names, quantiles=[0.16, 0.5, 0.84], show_titles=True
    )
    plt.suptitle(title)
    plt.tight_layout()
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
        cauchy_samples = scipy.stats.cauchy.rvs(size=size)
        normal_samples = scipy.stats.norm.rvs(size=size)

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
    # Save/plot figure
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path, dpi=300)
