import math
from src.distributions import simple_posterior


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
