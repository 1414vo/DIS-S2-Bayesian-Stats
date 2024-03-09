import scipy.stats as stats
import numpy as np


def simple_posterior(
    x, alpha, beta, alpha_min=-20, alpha_max=20, beta_min=0, beta_max=50
):
    """!Defines the non-normalized posterior for the simple detection problem.

    @param x            A set of observed light detections.
    @param alpha        The location of the lighthouse.
    @param beta         The height of the lighthouse.
    @param alpha_min    The lower bound of the lighthouse location.
    @param alpha_max    The upper bound of the lighthouse location.
    @param beta_min     The lower bound of the lighthouse height.
    @param beta_max     The upper bound of the lighthouse height.

    @returns        A posterior probability density for the location and height."""

    likelihood = stats.cauchy.pdf(x, loc=alpha, scale=beta)
    # Compute the 2 components of the prior
    prior_alpha = stats.uniform.pdf(alpha, loc=alpha_min, scale=alpha_max - alpha_min)
    prior_beta = stats.uniform.pdf(beta, loc=beta_min, scale=beta_max - beta_min)

    return likelihood * prior_alpha * prior_beta


def intensity_posterior(
    x,
    log_i,
    alpha,
    beta,
    i_0,
    alpha_min=-20,
    alpha_max=20,
    beta_min=0,
    beta_max=50,
    i_min=1e-5,
    i_max=100,
):
    """!Defines the non-normalized posterior for the simple detection problem.

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

    @returns            A posterior probability density for the location, height and intensity.
    """

    likelihood_location = stats.cauchy.logpdf(x, loc=alpha, scale=beta)
    d = beta**2 + (x - alpha) ** 2
    likelihood_intensity = stats.norm.logpdf(log_i, loc=np.log(i_0 / d), scale=1)

    # Compute the 3 components of the prior
    prior_alpha = stats.uniform.logpdf(
        alpha, loc=alpha_min, scale=alpha_max - alpha_min
    )
    prior_beta = stats.uniform.logpdf(beta, loc=beta_min, scale=beta_max - beta_min)
    prior_i = stats.loguniform.logpdf(i_0, a=i_min, b=i_max)

    return (
        likelihood_location + likelihood_intensity + prior_alpha + prior_beta + prior_i
    )
