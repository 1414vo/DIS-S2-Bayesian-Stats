import math
import numpy as np
from src.distributions import simple_posterior, intensity_posterior


def test_simple_posterior():
    """!Tests the posterior implementation"""
    result = simple_posterior(
        1, alpha=0, alpha_min=-0.5, alpha_max=0.5, beta=1, beta_min=0, beta_max=2
    )
    assert math.isclose(
        result, 1 / 4 / math.pi
    ), f"Posterior does not have the correct value, expected {1 / 4 / math.pi}, got {result}"

    assert (
        simple_posterior(0, alpha=-100, beta=1) == 0
    ), "Posterior does not behave correctly for out of range parameters"

    assert (
        simple_posterior(0, alpha=0, beta=1e4) == 0
    ), "Posterior does not behave correctly for out of range parameters"


def test_intensity_posterior():
    """!Tests the posterior implementation"""
    result = intensity_posterior(
        1,
        np.log(0.01),
        alpha=0,
        alpha_min=-0.5,
        alpha_max=0.5,
        beta=1,
        beta_min=0,
        beta_max=2,
        i_0=1,
        i_min=1e-5,
        i_max=100,
    )
    assert math.isclose(
        result, -13.8818673719
    ), f"Posterior does not have the correct value, expected {-13.881867}, got {result}"

    assert (
        intensity_posterior(0, np.log(0.01), alpha=-100, beta=1, i_0=1) == -np.inf
    ), "Posterior does not behave correctly for out of range parameters"

    assert (
        intensity_posterior(0, np.log(0.01), alpha=0, beta=1e4, i_0=1) == -np.inf
    ), "Posterior does not behave correctly for out of range parameters"

    assert (
        intensity_posterior(0, np.log(0.01), alpha=-100, beta=1, i_0=1e9) == -np.inf
    ), "Posterior does not behave correctly for out of range parameters"
