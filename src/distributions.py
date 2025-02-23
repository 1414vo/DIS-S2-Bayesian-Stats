"""!
@file   distributions.py
@brief  This file contains functions for computing the log-posterior of the lighthouse location
and intensity parameters based on observed data.

The file implements Bayesian inference techniques to estimate the location, height, and
intensity of a lighthouse.It includes functions for computing the log-likelihood, log-prior,
and log-posterior for the lighthouse detection problem.

@author Ivo Petrov
@date   13/03/2024
"""
import scipy.stats as stats
import numpy as np
from numpy.typing import ArrayLike


def simple_posterior(
    x: ArrayLike,
    alpha: float,
    beta: float,
    alpha_min: float = -20,
    alpha_max: float = 20,
    beta_min: float = 0,
    beta_max: float = 50,
) -> float:
    """!Defines the logarithm non-normalized posterior for the simple detection problem.

    @param x            A set of observed light detections.
    @param alpha        The location of the lighthouse.
    @param beta         The height of the lighthouse.
    @param alpha_min    The lower bound of the lighthouse location.
    @param alpha_max    The upper bound of the lighthouse location.
    @param beta_min     The lower bound of the lighthouse height.
    @param beta_max     The upper bound of the lighthouse height.

    @returns        A log of the posterior probability density for the location and height.
    """

    likelihood = np.sum(np.log(stats.cauchy.pdf(x, loc=alpha, scale=beta)))
    # Compute the 2 components of the prior
    prior_alpha = np.log(
        stats.uniform.pdf(alpha, loc=alpha_min, scale=alpha_max - alpha_min)
    )
    prior_beta = np.log(
        stats.uniform.pdf(beta, loc=beta_min, scale=beta_max - beta_min)
    )

    # Convert any nans to an equivalent of 0 probability
    return np.nan_to_num(likelihood + prior_alpha + prior_beta, nan=-np.inf)


def intensity_likelihood(
    x: ArrayLike, log_i: ArrayLike, alpha: float, beta: float, i_0: float
) -> float:
    """!Defines the log-likelihood for the detection problem,
    including information about the light intensity.

    @param x            A set of observed light detections.
    @param log_i        A set of observed log-intensities.
    @param alpha        The location of the lighthouse.
    @param beta         The height of the lighthouse.
    @param i_0          The intensity at the lighthouse origin.

    @returns            The log-likelihood for the location, height and intensity.
    """
    # Compute the likelihood for the location
    likelihood_location = stats.cauchy.logpdf(x, loc=alpha, scale=beta).sum()

    # Compute the likelihood for the intensity
    d = beta**2 + (x - alpha) ** 2
    likelihood_intensity = stats.norm.logpdf(log_i, loc=np.log(i_0 / d), scale=1).sum()

    return likelihood_location + likelihood_intensity


def intensity_posterior(
    x: ArrayLike,
    log_i: ArrayLike,
    alpha: float,
    beta: float,
    i_0: float,
    alpha_min: float = -20,
    alpha_max: float = 20,
    beta_min: float = 0,
    beta_max: float = 50,
    i_min: float = 1e-2,
    i_max: float = 10,
) -> float:
    """!Defines the non-normalized posterior for the detection problem,
    including information about the light intensity.

    @param x            A set of observed light detections.
    @param log_i        A set of observed log-intensities.
    @param alpha        The location of the lighthouse.
    @param beta         The height of the lighthouse.
    @param i_0          The intensity at the lighthouse origin.
    @param alpha_min    The lower bound of the lighthouse location.
    @param alpha_max    The upper bound of the lighthouse location.
    @param beta_min     The lower bound of the lighthouse height.
    @param beta_max     The upper bound of the lighthouse height.
    @param i_min        The lower bound of the lighthouse intensity.
    @param i_max        The upper bound of the lighthouse intensity.

    @returns            A log of the posterior probability density for the
    location, height and intensity.
    """
    likelihood = intensity_likelihood(x, log_i, alpha, beta, i_0)

    # Compute the 3 components of the prior
    prior_alpha = stats.uniform.logpdf(
        alpha, loc=alpha_min, scale=alpha_max - alpha_min
    )
    prior_beta = stats.uniform.logpdf(beta, loc=beta_min, scale=beta_max - beta_min)
    prior_i = stats.loguniform.logpdf(i_0, a=i_min, b=i_max)

    # Convert any nans to an equivalent of 0 probability
    return np.nan_to_num(likelihood + prior_alpha + prior_beta + prior_i, nan=-np.inf)
