"""!
@file   sampling.py
@brief  This file contains implementations of the Metropolis-Hastings algorithm, emcee sampler,
and a Nessai model for posterior sampling. It also includes utility functions for processing and
cleaning the generated Markov chains.

@author Ivo Petrov
@date   13/03/2024
"""
import numpy as np
import scipy.stats as stats
from emcee import EnsembleSampler
from emcee.autocorr import integrated_time
from nessai.model import Model
from nessai.flowsampler import FlowSampler


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

    # Initialize storage variables
    num_accept = 0
    samples = np.zeros((n_iter + 1, len(starting_point)))
    samples[0] = starting_point

    for i in range(n_iter):
        x_current = samples[i]

        # Sample from propsal distribution and compute the required a
        x_proposed = stats.multivariate_normal.rvs(x_current, cov_matrix)
        log_a = log_pdf(*x_proposed) - log_pdf(*x_current)

        # Determine whether to accept/reject
        u = np.random.uniform()
        if np.log(u) < log_a:
            samples[i + 1] = x_proposed
            num_accept += 1
        else:
            samples[i + 1] = x_current

    return samples, num_accept, num_accept / n_iter


def emcee_sampler(
    log_pdf, prior_distributions, n_iter, n_dim, n_walkers=100, random_seed=0
):
    """! A sampler that utilises the implementation from the "emcee" library.

    @param log_pdf              The logarithm of the target PDF.
    @param prior_distributions  A list for a prior function for each variable.
    @param n_iter               The number of iterations for the algorithm.
    @param n_dim                The number of dimensions for the sample space.
    @param n_walkers            The number of different Markov processes from which we sample.
    @param random_seed          The random seed.

    @returns                    The generated Markov chain."""
    np.random.seed(random_seed)
    starting_points = np.vstack(
        [
            prior_distributions[i].rvs(size=n_walkers, random_state=i)
            for i in range(len(prior_distributions))
        ]
    ).T

    sampler = EnsembleSampler(n_walkers, n_dim, log_pdf)
    sampler.run_mcmc(starting_points, n_iter)
    chains = sampler.get_chain()

    # Interleave the chains from all walkers
    return np.hstack([*chains]).reshape(-1, chains[0].shape[1])


class NessaiModel(Model):
    """! A nessai model from sampling from a posterior distribution given a prior and a likelihood.

    @param param_names          The names of the variables being sampled.
    @param param_bounds         A dictionary of bounds for each variable.
    @param prior_distributions  A dictionary for a prior function for each variable.
    @param likelihood           The likelihood of the data given the parameter set.
    """

    def __init__(self, param_names, param_bounds, prior_distributions, likelihood):
        """! Initializes the Nessai sampler.

        @param param_names          The names of the variables being sampled.
        @param param_bounds         A dictionary of bounds for each variable.
        @param prior_distributions  A dictionary for a prior function for each variable.
        @param likelihood           The likelihood function of the data given the parameter set.
        """
        self.names = param_names
        self.bounds = param_bounds
        self.prior_distributions = prior_distributions
        self.likelihood = likelihood

    def log_prior(self, x):
        """! The prior for the parameter set.
        @param      The array of parameters.
        @returns    The log-prior for the parameters.
        """
        log_p = np.log(self.in_bounds(x), dtype="float")
        # Sum up the prior distributions
        for name in self.names:
            log_p += np.log(self.prior_distributions[name](x[name]))
        return log_p

    def log_likelihood(self, x):
        """! The likelihood for the parameter set.
        @param      The array of parameters.
        @returns    The log-likelihood for the parameters.
        """
        # Apply the likelihood function
        return self.likelihood(**{name: x[name] for name in self.names})


def nessai_sampler(model, n_iter, random_seed=0, output_path="./"):
    """! An AI sampler for a predefined model.

    @param model        The defined Nessai model.
    @param n_iter       The number of iterations for the algorithm.
    @param random_seed  The random seed.
    @param output_path  The output path for automatic diagnostics of the sampler.

    @returns            The generated sample set."""
    sampler = FlowSampler(model, output=output_path, seed=random_seed, nlive=n_iter)
    sampler.run()
    samples = np.vstack([sampler.posterior_samples[param] for param in model.names]).T

    return samples


def clean_chain(chain, burnin=0):
    """! Cleans up the Markov Chain including for correlations and burnin.

    @param chain    The list of samples.
    @param burnin   The initial number of samples to be skipped.

    @returns        The cleaned up samples, ensured to likely be i.i.d."""
    # Compute integrated autocorrelation time
    tau = [integrated_time(chain[:, i]) for i in range(chain.shape[1])]
    thin = max(2 * int(np.max(tau)) - 1, 1)

    # Clean the chain for independent samples
    samples = chain[burnin::thin, :]

    print(f"Final number of samples is {len(samples)}")
    return samples
