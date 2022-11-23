import numpy as np
import math


def log_normal_density(x, mean, log_std, std):
    """returns guassian density given x on log scale"""

    variance = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * variance) - 0.5 *\
        np.log(2 * np.pi) - log_std    # num_env * frames * act_size
    log_density = log_density.sum(dim=-1, keepdim=True)  # num_env * frames * 1
    return log_density



