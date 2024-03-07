import scipy.stats as stats
import numpy as np


def simple_posterior(
    x, alpha, beta, alpha_min=-20, alpha_max=20, beta_min=0, beta_max=50
):
    """!Defines the logarithm non-normalized posterior for the simple detection problem.

    @param x        A set of observed light detections.
    @param alpha    The location of the lighthouse.
    @param beta     The height of the lighthouse.
    @param alpha_min    The lower bound of the lighthouse location.
    @param alpha_max    The upper bound of the lighthouse location.
    @param beta_min    The lower bound of the lighthouse height.
    @param beta_max    The upper bound of the lighthouse height.

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
    return np.nan_to_num(likelihood + prior_alpha + prior_beta, nan=-np.inf)
