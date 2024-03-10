import math
import numpy as np
from src.diagnostics import symmetric_kl_divergence


def test_sample_kl_divergence():
    # Use large samples for better estimation
    sample_1 = np.array([[1, 2, 2, 3, 3, 4, 5] * 500]).T
    sample_2 = np.array([[1, 2, 3, 4, 4, 4, 5] * 500]).T
    # Define true probabilities for discrete distribution
    ps1 = np.array([1 / 7, 2 / 7, 2 / 7, 1 / 7, 1 / 7])
    ps2 = np.array([1 / 7, 1 / 7, 1 / 7, 3 / 7, 1 / 7])

    # Test for identical samples
    res = symmetric_kl_divergence(sample_1, sample_1)
    for i in range(3):
        assert math.isclose(res[2 * i], 0), "KL divergence should be close to 0"
        assert math.isclose(res[2 * i + 1], 0), "KLD error should be close to 0"

    # Test for different samples (pre-computed values)
    # We check whether each results is in the 95% confidence interval (2 stds)
    # Note that the example uses a discrete distribution which is bound to be less accurate
    # than a continuous one for KDE estimation.
    true_kl_1 = np.sum(ps2 * np.log(ps2 / ps1))
    true_kl_2 = np.sum(ps1 * np.log(ps1 / ps2))
    res = symmetric_kl_divergence(sample_1, sample_2)
    assert res[0] - 2 * res[1] < true_kl_1 < res[0] + 2 * res[1]
    assert res[2] - 2 * res[3] < true_kl_2 < res[2] + 2 * res[3]
