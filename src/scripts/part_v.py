import numpy as np
import scipy.stats as stats
from src.diagnostics import distribution_summaries, chain_convergence_diagnostics
from src.plotting import autocorr_plot, corner_plot, trace_plot
from src.sampling import (
    metropolis_hastings,
    emcee_sampler,
    nessai_sampler,
    clean_chain,
    NessaiModel,
)
from src.distributions import simple_posterior


def execute_part_v(data_path: str, output_path: str):
    data = np.loadtxt(data_path)[:, 0]
    true_pdf = lambda params: simple_posterior(x=data, alpha=params[0], beta=params[1])
    log_pdf = lambda alpha, beta: simple_posterior(x=data, alpha=alpha, beta=beta)
    param_names = [r"$\alpha$", r"$\beta$"]

    # Metropolis-Hastings sampling (NOTE: Takes up to 10 minutes)
    # Use identity covariance for our proposal distribution
    print("Metropolis-Hastings")
    print("---------------------------")

    # Generate chain
    cov_matrix = np.eye(2)
    mh_chain = metropolis_hastings([0, 1], log_pdf, cov_matrix, n_iter=500000)

    # Summary statistics
    print(f"Accepted {mh_chain[1]} out of 500000 ({mh_chain[2] * 100: .1f}%)")
    mh_samples = clean_chain(mh_chain[0])
    chains = mh_chain[1:].reshape(10, 50000, 2)
    chain_convergence_diagnostics(chains, mh_samples, param_names)

    # Plots
    trace_plot(
        mh_chain,
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
        mh_chain,
        param_names,
        title="Autocorrelations in Metropolis-Hastings sampling",
        out_path=f"{output_path}/mh_autocorr.png",
    )

    # Emcee sampler
    print("EMCEE sampler")
    print("---------------------------")

    # Explicitly define distributions for sampling starting points
    starting_distributions = [
        stats.uniform(loc=-1, scale=2),
        stats.uniform(loc=1e-2, scale=10),
    ]

    # Generate chain
    emcee_chain = emcee_sampler(
        log_pdf, starting_distributions, n_iter=10000, n_dim=2, n_walkers=10
    )
    emcee_samples = clean_chain(emcee_chain)
    chains = emcee_chain[: len(emcee_chain) // 10 * 10].reshape(
        10, len(emcee_chain) // 10, 2
    )

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

    # Nessai Sampler
    print("Nessai sampler")
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
            stats.cauchy.logpdf(data[:, 0], loc=alpha, scale=beta)
        ),
    )
    # Generate chain
    nessai_chain = nessai_sampler(nessai_model, n_iter=10000)
    chains = nessai_chain[: len(nessai_chain) // 10 * 10].reshape(
        10, len(nessai_chain) // 10, 2
    )

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

    # Compute KL divergences for comparison of distributions
    distribution_summaries(
        samples=[mh_samples, emcee_samples, nessai_chain],
        algo_names=["Metropolis-Hastings", "Emcee", "Nessai"],
        true_pdf=true_pdf,
    )


if __name__ == "__main__":
    execute_part_v("../data/data.txt", "./out")
