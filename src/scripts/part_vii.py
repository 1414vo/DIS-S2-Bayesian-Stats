import numpy as np
import scipy.stats as stats
import argparse
import warnings
from src.diagnostics import distribution_summaries, chain_convergence_diagnostics
from src.plotting import autocorr_plot, corner_plot, trace_plot
from src.sampling import (
    metropolis_hastings,
    emcee_sampler,
    nessai_sampler,
    clean_chain,
    NessaiModel,
)
from src.distributions import intensity_posterior, intensity_likelihood


def execute_part_vii(data_path: str, output_path: str):
    # Suppress warnings (which are irrelevant to the program's execution)
    warnings.simplefilter("ignore")

    x = np.loadtxt(data_path)[:, 0]
    log_i = np.loadtxt(data_path)[:, 1]
    true_pdf = lambda params: intensity_posterior(
        x=x, log_i=log_i, alpha=params[0], beta=params[1], i_0=params[2]
    )
    log_pdf = lambda alpha, beta, i_0: intensity_posterior(
        x=x, log_i=log_i, alpha=alpha, beta=beta, i_0=i_0
    )
    param_names = [r"$\alpha$", r"$\beta$", r"$I_0$"]

    # Metropolis-Hastings sampling (NOTE: Takes up to 10 minutes)
    # Use identity covariance for our proposal distribution
    print("Metropolis-Hastings")
    print("---------------------------")

    # Generate chain
    cov_matrix = np.eye(3)
    mh_chain = metropolis_hastings([0, 1, 1], log_pdf, cov_matrix, n_iter=500000)

    # Summary statistics
    print(f"Accepted {mh_chain[1]} out of 500000 ({mh_chain[2] * 100: .1f}%)")
    mh_samples = clean_chain(mh_chain[0])

    # Print parameter estimates
    print(f"Estimate for alpha: {mh_samples[:, 0].mean()} +- {mh_samples[:, 0].std()}")
    print(f"Estimate for beta: {mh_samples[:, 1].mean()} +- {mh_samples[:, 1].std()}")
    print(f"Estimate for I_0: {mh_samples[:, 2].mean()} +- {mh_samples[:, 2].std()}")

    # Convergence diagnostic information
    chains = mh_chain[0][1:].reshape(10, 50000, 3)
    print(f"Number of samples: {len(mh_samples)}")
    chain_convergence_diagnostics(chains, mh_samples, param_names)

    # Plots
    trace_plot(
        mh_chain[0],
        param_names,
        title="Trace plot for Metropolis-Hastings sampling",
        out_path=f"{output_path}/mh_trace_w_intensity.png",
    )
    corner_plot(
        mh_samples,
        param_names,
        title="2D and 1D posterior for Metropolis-Hastings sampling",
        out_path=f"{output_path}/mh_corner_w_intensity.png",
    )
    autocorr_plot(
        mh_chain[0],
        param_names,
        title="Autocorrelations in Metropolis-Hastings sampling",
        out_path=f"{output_path}/mh_autocorr_w_intensity.png",
    )

    # Emcee sampler
    print("EMCEE sampler")
    print("---------------------------")

    # Explicitly define distributions for sampling starting points
    starting_distributions = [
        stats.uniform(loc=-1, scale=2),
        stats.uniform(loc=1e-2, scale=10),
        stats.loguniform(a=1e-5, b=100),
    ]

    # Generate chain
    emcee_chain = emcee_sampler(
        true_pdf, starting_distributions, n_iter=10000, n_dim=3, n_walkers=10
    )
    emcee_samples = clean_chain(emcee_chain)
    chains = emcee_chain[: len(emcee_chain) // 10 * 10].reshape(
        10, len(emcee_chain) // 10, 3
    )

    # Print parameter estimates
    print(
        f"Estimate for alpha: {emcee_samples[:, 0].mean()} +- {emcee_samples[:, 0].std()}"
    )
    print(
        f"Estimate for beta: {emcee_samples[:, 1].mean()} +- {emcee_samples[:, 1].std()}"
    )
    print(
        f"Estimate for I_0: {emcee_samples[:, 2].mean()} +- {emcee_samples[:, 2].std()}"
    )

    # Convergence diagnostic information
    print(f"Number of samples: {len(emcee_samples)}")
    chain_convergence_diagnostics(chains, emcee_samples, param_names)

    # Plots
    trace_plot(
        emcee_chain,
        param_names,
        title="Trace plot for Emcee Ensembler sampling",
        out_path=f"{output_path}/emcee_trace_w_intensity.png",
    )
    corner_plot(
        emcee_samples,
        param_names,
        title="2D and 1D posterior for Emcee Ensembler sampling",
        out_path=f"{output_path}/emcee_corner_w_intensity.png",
    )
    autocorr_plot(
        emcee_chain,
        param_names,
        title="Autocorrelations in Emcee Ensembler sampling",
        out_path=f"{output_path}/emcee_autocorr_w_intensity.png",
    )

    # Nessai Sampler
    print("Nessai sampler")
    print("---------------------------")

    # The Nessai model requires us to define the priors and likelihood explicitly instead of being given
    # the unnormalized posterior
    nessai_model = NessaiModel(
        param_names=["alpha", "beta", "i_0"],
        param_bounds={"alpha": [-20, 20], "beta": [0, 50], "i_0": [1e-5, 100]},
        prior_distributions={
            "alpha": stats.uniform(loc=-20, scale=40).pdf,
            "beta": stats.uniform(loc=0, scale=50).pdf,
            "i_0": stats.loguniform(a=1e-5, b=100).pdf,
        },
        likelihood=lambda alpha, beta, i_0: intensity_likelihood(
            x=x, log_i=log_i, alpha=alpha, beta=beta, i_0=i_0
        ),
    )
    # Generate chain
    nessai_chain = nessai_sampler(
        nessai_model, n_iter=10000, output_path=f"{output_path}/nessai_p7"
    )
    chains = nessai_chain[: len(nessai_chain) // 10 * 10].reshape(
        10, len(nessai_chain) // 10, 3
    )

    # Print parameter estimates
    print(
        f"Estimate for alpha: {nessai_chain[:, 0].mean()} +- {nessai_chain[:, 0].std()}"
    )
    print(
        f"Estimate for beta: {nessai_chain[:, 1].mean()} +- {nessai_chain[:, 1].std()}"
    )
    print(
        f"Estimate for I_0: {nessai_chain[:, 2].mean()} +- {nessai_chain[:, 2].std()}"
    )

    # Convergence diagnostic information
    print(f"Number of samples: {len(nessai_chain)}")
    chain_convergence_diagnostics(chains, nessai_chain, param_names)

    # Plots
    trace_plot(
        nessai_chain,
        param_names,
        title="Trace plot for Nessai sampling",
        out_path=f"{output_path}/nessai_trace_w_intensity.png",
    )
    corner_plot(
        nessai_chain,
        param_names,
        title="2D and 1D posterior for Nessai sampling",
        out_path=f"{output_path}/nessai_corner_w_intensity.png",
    )
    autocorr_plot(
        nessai_chain,
        param_names,
        title="Autocorrelations in Nessai sampling",
        out_path=f"{output_path}/nessai_autocorr_w_intensity.png",
    )

    # Compute KL divergences for comparison of distributions
    distribution_summaries(
        samples=[mh_samples, emcee_samples, nessai_chain],
        algo_names=["Metropolis-Hastings", "Emcee", "Nessai"],
        true_pdf=true_pdf,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Part VII Parameter Estimation",
        description="Estimates the lighthouse location parameters using different MCMC algorithms.",
    )
    parser.add_argument("data_path", help="Location of the data file.")
    parser.add_argument("out_path", help="Location of the output folder.")

    args = parser.parse_args()

    execute_part_vii(args.data_path, args.out_path)
