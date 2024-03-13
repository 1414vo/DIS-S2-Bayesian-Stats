"""!
@file   part_v.py
@brief  Executes parameter estimation using Metropolis-Hastings, emcee, and Nessai sampling methods.

This script is designed to perform parameter estimation on lighthouse location data using different
Markov Chain Monte Carlo (MCMC) algorithms.

@author Ivo Petrov
@date   13/03/2024
"""
import numpy as np
import scipy.stats as stats
import argparse
import warnings
from src.diagnostics import distribution_summaries, chain_convergence_diagnostics
from src.plotting import autocorr_plot, corner_plot, trace_plot, compare_prior
from src.sampling import (
    metropolis_hastings,
    emcee_sampler,
    nessai_sampler,
    clean_chain,
    NessaiModel,
)
from src.distributions import simple_posterior


def execute_part_v(data_path: str, output_path: str, do_kld: bool):
    """! Sequentially executes the Metropolis-Hastings, Ensemble Sampling and
    Nested Sampling algorithms alongside all diagonstic plots and measurements.
    Uses the information from the light detection's location only.

    @param data_path    The location of the data file.
    @param output_path  Where to store the plots/diagnostic information.
    @param do_kld       Whether to generate the Kullback-Leibler
    information (NOTE: computationally expensive).
    """
    # Suppress warnings (which are irrelevant to the program's execution)
    warnings.simplefilter("ignore")

    data = np.loadtxt(data_path)[:, 0]
    true_pdf = lambda params: simple_posterior(x=data, alpha=params[0], beta=params[1])
    log_pdf = lambda alpha, beta: simple_posterior(x=data, alpha=alpha, beta=beta)
    param_names = [r"$\alpha$", r"$\beta$"]
    prior_distributions = [
        stats.uniform(loc=-20, scale=40),
        stats.uniform(loc=0, scale=50),
    ]
    x_ranges = [
        [-4, 4],
        [0, 6],
    ]
    y_scales = ["linear", "linear"]

    # Metropolis-Hastings sampling (NOTE: Takes up to 10 minutes)
    # Use identity covariance for our proposal distribution
    print("Metropolis-Hastings")
    print("---------------------------")

    # Generate chain
    cov_matrix = np.eye(2)
    mh_chain = metropolis_hastings([0, 1], log_pdf, cov_matrix, n_iter=100000)

    # Summary statistics
    print(f"Accepted {mh_chain[1]} out of 100000 ({mh_chain[2] * 100: .1f}%)")
    mh_samples = clean_chain(mh_chain[0])

    # Print parameter estimates
    print(f"Esitmate for alpha: {mh_samples[:, 0].mean()} +- {mh_samples[:, 0].std()}")
    print(f"Esitmate for beta: {mh_samples[:, 1].mean()} +- {mh_samples[:, 1].std()}")

    # Convergence diagnostic information
    chains = mh_chain[0][1:].reshape(10, 10000, 2)
    print(f"Number of samples: {len(mh_samples)}")
    chain_convergence_diagnostics(chains, mh_samples, param_names)

    # Plots
    trace_plot(
        mh_chain[0],
        param_names,
        title="Trace plot for Metropolis-Hastings sampling",
        out_path=f"{output_path}/mh_trace.png",
    )
    corner_plot(
        mh_samples,
        param_names,
        title="2D and 1D posterior for Metropolis-Hastings sampling",
        out_path=f"{output_path}/mh_corner.png",
    )
    autocorr_plot(
        mh_chain[0],
        param_names,
        max_lag=200,
        title="Autocorrelations in Metropolis-Hastings sampling",
        out_path=f"{output_path}/mh_autocorr.png",
    )
    compare_prior(
        prior_distributions,
        mh_samples.T,
        param_names,
        x_ranges,
        y_scales,
        out_path=f"{output_path}/mh_prior_comparison.png",
    )

    # Emcee sampler
    print("\nEMCEE sampler")
    print("---------------------------")

    # Explicitly define distributions for sampling starting points
    starting_distributions = [
        stats.uniform(loc=-1, scale=2),
        stats.uniform(loc=1e-2, scale=10),
    ]

    # Generate chain
    emcee_chain = emcee_sampler(
        true_pdf, starting_distributions, n_iter=50000, n_dim=2, n_walkers=10
    )
    emcee_samples = clean_chain(emcee_chain)
    chains = emcee_chain[: len(emcee_chain) // 10 * 10].reshape(
        10, len(emcee_chain) // 10, 2
    )

    # Print parameter estimates
    print(
        f"Esitmate for alpha: {emcee_samples[:, 0].mean()} +- {emcee_samples[:, 0].std()}"
    )
    print(
        f"Esitmate for beta: {emcee_samples[:, 1].mean()} +- {emcee_samples[:, 1].std()}"
    )

    # Convergence diagnostic information
    print(f"Number of samples: {len(emcee_samples)}")
    chain_convergence_diagnostics(chains, emcee_samples, param_names)

    # Plots
    trace_plot(
        emcee_chain,
        param_names,
        title="Trace plot for Emcee Ensembler sampling",
        out_path=f"{output_path}/emcee_trace.png",
    )
    corner_plot(
        emcee_samples,
        param_names,
        title="2D and 1D posterior for Emcee Ensembler sampling",
        out_path=f"{output_path}/emcee_corner.png",
    )
    autocorr_plot(
        emcee_chain,
        param_names,
        title="Autocorrelations in Emcee Ensembler sampling",
        out_path=f"{output_path}/emcee_autocorr.png",
    )
    compare_prior(
        prior_distributions,
        emcee_samples.T,
        param_names,
        x_ranges,
        y_scales,
        out_path=f"{output_path}/emcee_prior_comparison.png",
    )

    # Nessai Sampler
    print("\nNessai sampler")
    print("---------------------------")

    # The Nessai model requires us to define the priors and likelihood explicitly instead of being given
    # the unnormalized posterior
    nessai_model = NessaiModel(
        param_names=["alpha", "beta"],
        param_bounds={"alpha": [-20, 20], "beta": [0, 50]},
        prior_distributions={
            "alpha": stats.uniform(loc=-20, scale=40).pdf,
            "beta": stats.uniform(loc=0, scale=50).pdf,
        },
        likelihood=lambda alpha, beta: np.sum(
            stats.cauchy.logpdf(data, loc=alpha, scale=beta)
        ),
    )
    # Generate chain
    nessai_chain = nessai_sampler(
        nessai_model, n_iter=1000, output_path=f"{output_path}/nessai_p5"
    )
    chains = nessai_chain[: len(nessai_chain) // 10 * 10].reshape(
        10, len(nessai_chain) // 10, 2
    )

    # Print parameter estimates
    print(
        f"Esitmate for alpha: {nessai_chain[:, 0].mean()} +- {nessai_chain[:, 0].std()}"
    )
    print(
        f"Esitmate for beta: {nessai_chain[:, 1].mean()} +- {nessai_chain[:, 1].std()}"
    )

    # Convergence diagnostic information
    print(f"Number of samples: {len(nessai_chain)}")
    chain_convergence_diagnostics(chains, nessai_chain, param_names)

    # Plots
    trace_plot(
        nessai_chain,
        param_names,
        title="Trace plot for Nessai sampling",
        out_path=f"{output_path}/nessai_trace.png",
    )
    corner_plot(
        nessai_chain,
        param_names,
        title="2D and 1D posterior for Nessai sampling",
        out_path=f"{output_path}/nessai_corner.png",
    )
    autocorr_plot(
        nessai_chain,
        param_names,
        title="Autocorrelations in Nessai sampling",
        out_path=f"{output_path}/nessai_autocorr.png",
    )
    compare_prior(
        prior_distributions,
        nessai_chain.T,
        param_names,
        x_ranges,
        y_scales,
        out_path=f"{output_path}/nessai_prior_comparison.png",
    )
    print("\n")
    # Compute KL divergences for comparison of distributions
    distribution_summaries(
        samples=[mh_samples, emcee_samples, nessai_chain],
        algo_names=["Metropolis-Hastings", "Emcee", "Nessai"],
        do_kld=do_kld,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Part V Parameter Estimation",
        description="Estimates the lighthouse location parameters using different MCMC algorithms.",
    )
    parser.add_argument("data_path", help="Location of the data file.")
    parser.add_argument("out_path", help="Location of the output folder.")
    parser.add_argument("--kld", dest="kld", default=False, action="store_true")
    args = parser.parse_args()

    execute_part_v(args.data_path, args.out_path, args.kld)
