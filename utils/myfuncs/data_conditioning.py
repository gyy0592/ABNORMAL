

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from .LIGO_data import RunningMedian #, RunningMedian_gpu


def dumpNotchIIR_v3(straindata, stop_bands, stop_bandFactors, sample_rate=4096, numtaps=501, gstop_threshold=5):
    transient = numtaps + 1
    input_data = straindata.copy()
    output = straindata.copy()
    nyquist = sample_rate / 2
    gpass = 1  # Fixed passband ripple
    
    for lp, stop_band in enumerate(stop_bands):
        center_freq = np.mean(stop_band)
        band_width = max(stop_band[1] - stop_band[0], 1)
        stop_band = (center_freq - band_width / 2, center_freq + band_width / 2)
        pass_band = (center_freq - band_width * 0.75, center_freq + band_width * 0.75)
        Wp = np.array(pass_band) / nyquist
        Ws = np.array(stop_band) / nyquist

        target_gstop = -20 * np.log10(1/(stop_bandFactors[lp]))
        remaining_gstop = target_gstop

        while remaining_gstop > 0:
            # current_gstop = min(gstop_threshold, remaining_gstop)
            # current_gstop = max(gpass, current_gstop)
            current_gstop = min(gstop_threshold, remaining_gstop)
            current_gstop = max(gpass, current_gstop)
            N, Wn = signal.buttord(Wp, Ws, gpass, current_gstop)
            if N == 0:
                break
            b, a = signal.butter(N, Wn, btype='stop')
            output = signal.filtfilt(b, a, input_data)
            if np.isnan(output).any():
                raise ValueError("NaN values detected in the output.")

            remaining_gstop -= current_gstop
            input_data = output.copy()
            # remaining_gstop -= current_gstop
 

    return output[transient:-transient]

def obtain_smooth_asd(asd):
    """
    Process the ASD using two resolutions with weighted transition.
    
    Parameters:
        asd (numpy.ndarray): Input ASD array.
    
    Returns:
        numpy.ndarray: Processed ASD array with smooth transition between resolutions.
    """
    # Process with both window sizes
    high_res = obtain_smooth_asd_legacy(asd, window_size=1000)
    low_res = obtain_smooth_asd_legacy(asd, window_size=4500)
    
    # Define the boundary points and transition width
    edge_points = 1229
    transition_width = 200  # Width of transition region
    
    # Create the output array
    output = np.zeros_like(asd)
    
    # Create smooth transition weights
    transition_weights = np.cos(np.linspace(0, np.pi/2, transition_width))**2
    
    # Fill the first section (pure high res)
    output[:edge_points-transition_width] = high_res[:edge_points-transition_width]
    
    # Create weighted transition for the first boundary
    for i in range(transition_width):
        idx = edge_points - transition_width + i
        w = transition_weights[i]
        output[idx] = w * high_res[idx] + (1-w) * low_res[idx]
    
    # Fill the middle with pure low res
    output[edge_points:] = low_res[edge_points:]
    
    # Fill the last section (pure high res)
    output[-edge_points:] = high_res[-edge_points:]
    
    # Create weighted transition for the last boundary
    for i in range(transition_width):
        idx = -edge_points + i
        w = transition_weights[i]
        output[idx] = w * low_res[idx] + (1-w) * high_res[idx]
    
    return output

def obtain_smooth_asd_legacy(asd, window_size):
    """
    Process the ASD by applying a running median and interpolating it.

    Parameters:
        asd (numpy.ndarray): Input ASD array.
        window_size (int): Window size for the running median.

    Returns:
        numpy.ndarray: Processed ASD array, same length as the original.
    """
    # Apply running median
    asd_smoothed = RunningMedian(asd, window_size)
    
    # Interpolate to match the original length
    x_original = np.arange(len(asd))
    x_smoothed = np.arange(len(asd_smoothed))
    interpolator = interp1d(x_smoothed, asd_smoothed, kind='linear', fill_value='extrapolate')
    asd_interpolated = interpolator(x_original)
    
    return asd_interpolated