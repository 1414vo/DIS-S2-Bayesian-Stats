import arviz
import scipy
import numpy as np


def chain_convergence_diagnostics(chains, samples, param_names):
    """! Computes certain diagnostic statistics for the Markov chain and
    the extract samples"""
    # Create Arziv dataset
    dataset = {param: chains[:, :, i] for i, param in enumerate(param_names)}

    # Simple summary
    print(
        "Effective number of samples is {frac: .2f}% of the total samples".format(
            frac=len(samples) / chains.shape[0] / chains.shape[1] * 100
        )
    )

    # Compute Gelman-Rubic Diagnostic (R hat)
    r_hats = arviz.diagnostics.rhat(dataset)
    for param in param_names:
        print(
            f'Measured Gelman-Rubin statistic for parameter "{param}": {r_hats[param]}'
        )


def symmetric_kl_divergence(sample_1, sample_2):
    """ """
    # Approximate distributions using Kernel Density Estimation
    dist1_estimate = scipy.stats.gaussian_kde(sample_1.T)
    dist2_estimate = scipy.stats.gaussian_kde(sample_1.T)

    # Monte carlo integration for KL divergence

    kl_pq = np.mean(
        np.log(dist1_estimate(sample_1.T) + 1e-10)
        - np.log(dist2_estimate(sample_1.T) + 1e-10)
    )

    kl_pq_err = np.std(
        np.log(dist1_estimate(sample_1.T) + 1e-10)
        - np.log(dist2_estimate(sample_1.T) + 1e-10)
    ) / np.sqrt(len(sample_1))

    kl_qp = np.mean(
        np.log(dist2_estimate(sample_2.T) + 1e-10)
        - np.log(dist1_estimate(sample_2.T) + 1e-10)
    )

    kl_qp_err = np.std(
        np.log(dist2_estimate(sample_2.T) + 1e-10)
        - np.log(dist1_estimate(sample_2.T) + 1e-10)
    ) / np.sqrt(len(sample_2))

    return (kl_pq + kl_qp) / 2, np.sqrt(kl_pq_err**2 + kl_qp_err**2) / 2


def kl_divergence(sample, pdf):
    dist1_estimate = scipy.stats.gaussian_kde(sample.T)
    kl_pq = np.mean(
        np.log(dist1_estimate(sample.T) + 1e-10) - np.log(pdf(sample) + 1e-10)
    )

    kl_pq_err = np.std(
        np.log(dist1_estimate(sample) + 1e-10) - np.log(pdf(sample) + 1e-10)
    ) / np.sqrt(len(sample))

    return kl_pq, kl_pq_err
