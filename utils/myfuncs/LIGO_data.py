

import numpy as np
from scipy.ndimage import median_filter



def RunningMedian(x, N):
    """
    Calculate the running median of an array.

    Parameters:
        x (numpy.ndarray): Input data array.
        N (int): Window size for the running median.

    Returns:
        numpy.ndarray: Array of running medians.
    """
    pad = N // 2
    padded_x = np.pad(x[x != 0], (pad, pad), 'edge')
    return median_filter(padded_x, N)[pad:-pad]