import numpy as np
import scipy.stats as stats


def metropolis_hastings(
    starting_point, log_pdf, cov_matrix, n_iter=500000, random_seed=0
):
    """! Implements the Metropolis-Hastings algorithm with a Gaussian proposal
    distribution.

    @param starting_point   The initialization point for the algorithm.
    @param log_pdf          The logarithm of the target PDF.
    @param cov_matrix       The covariance matrix for the proposal distribution.
    Must be positive-definite.
    @param n_iter           The number of iterations for the algorithm. Defaults to 500,000.
    @param random_seed      The random seed.

    @returns                The generated Markov chain.
    """
    np.random.seed(random_seed)
    num_accept = 0
    samples = np.zeros((n_iter + 1, len(starting_point)))
    samples[0] = starting_point

    for i in range(n_iter):
        x_current = samples[i]
        x_proposed = stats.multivariate_normal(x_current, cov_matrix)
        log_a = log_pdf(*x_proposed) - log_pdf(*x_current)

        u = np.random.uniform()
        if np.log(u) < log_a:
            samples[i + 1] = x_proposed
            num_accept += 1
        else:
            samples[i + 1] = x_current

    return samples, num_accept, num_accept / n_iter
