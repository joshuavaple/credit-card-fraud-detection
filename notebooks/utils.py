import numpy as np


def compute_lognormal_parameters(mean: float, cv: float):
    """
    Calculate mu and sigma for lognormal distribution given mean and cv.
    Adapted from: https://www.johndcook.com/blog/2022/02/24/find-log-normal-parameters/

    """
    variance = (mean * cv) ** 2
    sigma2 = np.log(1 + variance / (mean**2))
    sigma = np.sqrt(sigma2)
    mu = np.log(mean) - sigma2 / 2
    return (mu.item(), sigma.item())
