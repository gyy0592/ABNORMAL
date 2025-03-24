

import numpy as np 
from scipy.signal import firwin
import cupy as cp
from ssqueezepy import cwt, icwt, Wavelet
from scipy.interpolate import interp1d

def generate_subbandsv2(freq_start, freq_end, nFreq, overlap_ratio, extra_bandwidth=12):
    # for numtaps = 1001 we have extra width of 12
    """
    Generate frequency sub-bands with overlapping regions, covering the entire frequency range.

    Parameters:
    freq_start (float): Start frequency of the range.
    freq_end (float): End frequency of the range.
    nFreq (int): Number of frequency bands to generate.
    overlap_ratio (float): Overlap ratio between consecutive bands.
    extra_bandwidth (float): Extra bandwidth added to account for filter response.

    Returns:
    numpy.ndarray: Array containing start and end frequencies of each sub-band.
    """
    # Calculate the width of each band, including the overlap
    total_width = freq_end - freq_start
    band_width_with_overlap = total_width / (nFreq - (nFreq - 1) * overlap_ratio)

    # Initialize the array to store sub-band frequencies
    subBands = np.zeros((nFreq, 2))

    for i in range(nFreq):
        if i == 0:
            # The first band starts at the start frequency
            start_freq = freq_start
        else:
            # Subsequent bands start at the end of the previous band minus the overlap
            start_freq = subBands[i - 1, 1] - band_width_with_overlap * overlap_ratio

        end_freq = start_freq + band_width_with_overlap
        subBands[i, 0] = start_freq
        subBands[i, 1] = end_freq
    subBands_real = subBands.copy()
    # Modify the subBands array to account for extra_bandwidth by narrowing the bands
    for i in range(nFreq):
        current_bandwidth = subBands[i, 1] - subBands[i, 0]
        if extra_bandwidth >= current_bandwidth:
            raise ValueError(f"Extra bandwidth {extra_bandwidth} is larger than or equal to the current band width {current_bandwidth}.")

        # Narrow the band by reducing the width from both ends
        subBands[i, 0] += extra_bandwidth / 2
        subBands[i, 1] -= extra_bandwidth / 2
    subBands4FIR = subBands.copy()
    return subBands4FIR, subBands_real


def beta_extract_func(time_series_data, nFreq=100, fs=4096, overlap_freq=0.75, means= None, std_devs= None):
    freq_start = 30 
    freq_end = 1700 
    # Generating subBands array
    subBands4FIR, subBands = generate_subbandsv2(freq_start, freq_end, nFreq, overlap_freq, 12)
    # Bandpass filtering using subBanddds

    numtaps = 1002
    # bandpassed_data_noUnderSamp = [bandpass_filter(time_series_data, subBands[i, 0], subBands[i, 1], fs, numtaps) for i in range(nFreq)]
    bandpassed_data_noUnderSamp = [bandpass_filter_gpu(time_series_data, subBands4FIR[i, 0], subBands4FIR[i, 1], fs, numtaps) for i in range(nFreq)]
    
    bandpassed_data = generate_undersampled_datav2(bandpassed_data_noUnderSamp, subBands, fs, numtaps)
    max_length = max(len(data) for data in bandpassed_data[int(nFreq/4):])
    # current version is 1250
    bandpassed_data = resampling(bandpassed_data,max(max_length,1000))
    # bandpassed_data = resampling(bandpassed_data,1000)

    # Initialize results
    histograms = []
    # running_variance_window = new_sample_interval
    # running_variance_stride = new_sample_interval//2
    series_length = 20
    # dataLen = len(heterodyned_data[0])
    # series_length = 1 + (dataLen - running_variance_window) // running_variance_stride
    # series_length = 64
    if np.isnan(time_series_data).any():
        temp_hist = np.zeros((series_length, nFreq+1))
        temp_hist[:,:]=np.nan
        return temp_hist
    
    # running_variance = running_statistic(normalized_timeseries_data, series_length)
    # running_variance = running_variance / np.linalg.norm(running_variance)
    running_stat = []
    running_means, running_stds = [], []
    multi_chanData = []
    prop = 0.4
    propLeft = (1 - prop) / 2
    # hist_data = []
    for lp, data in enumerate(bandpassed_data):
    # for data in heterodyned_data:
    # Iterate on both heterodyned data and bandpassed_data_noUnderSamp data

        # for ori_data, data in zip(bandpassed_data, heterodyned_data):
        # If mean is not None perform this, else use the original data
        if means is not None:
            mean = means[lp]
            std_dev = std_devs[lp]
            # normalized_data = (data - mean) / std_dev + mean
            normalized_data = data/std_dev
            isNormalized = True
        else: 
            running_variance = running_statistic(data, len(data)//500)
            std_dev = np.median(np.sqrt(running_variance))

            normalized_data = (data) / std_dev 

        if np.isnan(normalized_data).any():
            temp_hist = np.zeros((series_length, nFreq+1))
            temp_hist[:,:]=np.nan
            ValueError('Nan value detected')
            # return temp_hist
        running_std = running_statistic(normalized_data, series_length, statistic_func= np.std)
        running_mean = running_statistic(normalized_data, series_length,  statistic_func=np.mean)
        # normalize the data 
        # running_variance/=np.linalg.norm(running_variance)
        # running_mean /= np.linalg.norm(running_mean)


        running_stds.append(running_std)
        running_means.append(running_mean)

        # Histogram calculation
        counts, _ = np.histogram(normalized_data, bins=series_length)
        counts = np.array(counts).astype(np.float64)
        counts /= np.linalg.norm(counts)
        # print(counts.shape)
        histograms.append(np.array(counts))
        

    # Combine histograms and moments for each channel

    # reshape the histgoram, running mean/var into 1x n series and then concatenate them
    # original shape should be 100 x 32 x 3 -> 300 x 32  
    # reshape will be n x 32*k, set k to 5 the result will be 
    # histograms, running_means, running_stds= np.array(histograms).reshape(-1,32*10), np.array(running_means).reshape(-1,32*10), np.array(running_stds).reshape(-1,32*10)

    
    # combined_results = np.concatenate((histograms,running_means,running_stds), axis=0).T #20 x 300
    # combined_results = combined_results.reshape(-1,15) 
    combined_results = np.concatenate((histograms,running_means,running_stds), axis=0).reshape(15,-1).T #20 x 300
    return combined_results

def beta_wavelet(time_series_data, nFreq=5, fs=4096, overlap_freq=0.5, means= None, std_devs= None):
    numtaps= 1002

    # Calculating the width of each band including the overlap
    # bandpassed_data = wavelet_bandpass(time_series_data,nFreq,sampling_rate=fs, wavelet_basis='dmey',numtaps= numtaps)
    bandpassed_data = wavelet_bandpass(time_series_data,nFreq, numtaps= numtaps)
    # Initialize results
    # bandpassed_data = generate_undersampled_data(bandpassed_data_noUnderSamp, subBands, fs)
    histograms = []
    moments = []

    prop = 0.4
    propLeft = (1 - prop) / 2
    dataLen = len(bandpassed_data[0])

    # Calculate series_length from num_segments and num_bins
    # series_length = num_segments * num_bins
    series_length = int(bandpassed_data[-1].shape[0]/5)
    series_length = 20

    
    running_stat = []

    running_means, running_stds = [], []
    for lp, data in enumerate(bandpassed_data):

        # Data normalization
        if means is not None:
            mean = means[lp]
            std_dev = std_devs[lp]
            # normalized_data = (data - mean) / std_dev + mean
            normalized_data = (data) / std_dev
            
            isNormalized = True
        else: 
            normalizing_factor = np.median(np.abs(data))
            test_data = data / normalizing_factor
            # running_variance = running_statistic(data, 64)
            running_variance = running_statistic(test_data, series_length)
            std_dev = np.median(np.sqrt(running_variance) * normalizing_factor)


            normalized_data = (data) / std_dev
            # running_variance = running_variance/(std_dev**2)

        if np.isnan(normalized_data).any():
                temp_hist = np.zeros((series_length, nFreq+1))
                temp_hist[:,:]=np.nan
                # raise error: value error 
                ValueError('Nan value detected')
                # return temp_hist

        running_std = running_statistic(normalized_data, series_length, statistic_func=np.std)
        running_mean = running_statistic(normalized_data, series_length,  statistic_func=np.mean)
        # normalize the data 
        # running_variance/=np.linalg.norm(running_variance)
        # running_mean /= np.linalg.norm(running_mean)
        running_stds.append(running_std) 
        running_means.append(running_mean)
        # Histogram calculation
        # No segmentation 
        counts, _ = np.histogram(normalized_data, bins=series_length)
        # print(counts.shape)
        counts = np.array(counts).astype(np.float64) 
        counts /= np.linalg.norm(counts)
        histograms.append(counts)
        # 15 x 64
    
    combined_results = np.concatenate((histograms,running_means,running_stds), axis=0).T # 20 x 15
    # combined_results = combined_results.T
    return combined_results



def bandpass_filter_gpu(data, lowcut, highcut, fs, numtaps=6):
    # Calculate Nyquist frequency
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    transient_length = (numtaps + 1) // 2
    transient_length = numtaps + 1
    
    # Use SciPy to create FIR coefficients (not available in CuPy)
    fir_coeff = firwin(numtaps, [low, high], pass_zero=False)

    # Move FIR coefficients to GPU
    fir_coeff_gpu = cp.asarray(fir_coeff)

    # Convert data to CuPy array for GPU computation
    data_gpu = cp.asarray(data)

    # Perform convolution for the forward pass
    filtered_data_gpu = cp.convolve(data_gpu, fir_coeff_gpu, mode='same')

    # Perform convolution again for the backward pass to mimic filtfilt
    filtered_data_gpu = cp.convolve(filtered_data_gpu[::-1], fir_coeff_gpu, mode='same')[::-1]

    # Convert back to NumPy array if necessary
    output_data = cp.asnumpy(filtered_data_gpu)[transient_length:-transient_length]

    # Free GPU memory
    # del fir_coeff_gpu, data_gpu, filtered_data_gpu
    # cp.cuda.Stream.null.synchronize()  # Ensure all operations are complete
    # gc.collect()  # Run garbage collection

    # return output_data

    return output_data

def generate_undersampled_datav2(bandpassed_datas, subBands, fs, numtaps):
    """
    Generate undersampled data from bandpassed data.

    Parameters:
    bandpassed_datas (numpy.ndarray): Original data array.
    subBands (numpy.ndarray): Array of sub-band frequencies.
    fs (int): Original sampling rate.

    Returns:
    list: List of undersampled data and reconstructed data corresponding to each bandpass filter.
    """
    undersampled_data_list = []

    for lp, band in enumerate(subBands):
        # Apply bandpass filter
        bandpassed_data = bandpassed_datas[lp]
        N = find_maxN(fs,band)
        # undersampled_data = decimate(bandpassed_data, int(fs/new_nyquist_rate), ftype='fir')
        undersampled_data = bandpassed_data[::N]
        undersampled_data_list.append(undersampled_data)

    return undersampled_data_list


def running_statistic(data, series_length, statistic_func=np.var):
    data_length = len(data)
    if data_length < series_length:
        raise ValueError("Data length must be larger than the series length.")

    # Calculate initial window size based on the series length
    window_size = int(data_length / series_length)
    
    # Determine how many windows of size `window_size` and `window_size + 1` are needed

    windows_size_plus_one = data_length - series_length*window_size
    # Alternate between window sizes to cover the entire data
    result = []
    index = 0
    temp = 0
    for i in range(series_length):
        if i < windows_size_plus_one:
            current_window_size = window_size + 1
        else:
            current_window_size = window_size
        
        window = data[index:index + current_window_size]
        result.append(statistic_func(window))
        index += current_window_size
    #     temp +=current_window_size
    # print(temp)
    return np.array(result)


def find_maxN(fs,freqBand):
    # first fs/N>=freqBand[1]-freqBand[0]
    N_upper = np.floor(fs/(freqBand[1]-freqBand[0]))
    # fs/N/2 * k>=freqBand[1]    ->     k>= 2*freqBand[1]*N/fs
    # k_lower = np.ceil(2*freqBand[1]*N_upper/fs)
    # fs/N/2 * (k-1)<=freqBand[0]   ->   k <= 2*freqBand[0]/fs*N + 1 
    for N in range(int(N_upper)+1, 1, -1):  # Iterate from N_upper to 1
        if np.floor(2*freqBand[0]/fs*N + 1 ) >= 2*freqBand[1]*N/fs:
            break
        else:
            continue
    return N

def wavelet_bandpass(data, nFreq, wavelet_basis='morlet', numtaps = 6):
    # Determine the appropriate wavelet
    wavelet = Wavelet(wavelet_basis)

    # Compute CWT coefficients and scales
    # scales = np.geomspace(6, 256, nFreq*4)
    scales = np.geomspace(6.4, 224, nFreq*4)
    # scales = np.linspace(4,256,nFreq*2)
    # scales = np.geomspace(7, 180, nFreq*4)
    Wx, scales = cwt(data, wavelet=wavelet,scales = scales)

    # Number of scales
    num_scales = len(scales)

    # Divide scales into bands
    scale_indices = np.array_split(np.arange(num_scales), nFreq)

    # Reconstruct signals for each band
    reconstructed_signals = []
    transient_length = numtaps+1
    for indices in scale_indices:
        # Create a copy of Wx for band-specific modification
        Wx_band = np.zeros_like(Wx)
        
        # Set only the current band's indices to be non-zero
        Wx_band[indices, :] = Wx[indices, :]
        
        # Inverse CWT to reconstruct the band-specific signal
        reconstructed_signal = icwt(Wx_band, wavelet=wavelet,scales = scales)

        reconstructed_signal = reconstructed_signal[transient_length:-transient_length]
        if np.isnan(reconstructed_signal).any():
            print(reconstructed_signal[:10])
        # print the continous zero
        if np.isclose(reconstructed_signal,0).all():
            # print('close zero')
            pass
            # print(scale_indices[-5:])
        reconstructed_signals.append(reconstructed_signal)
    # print('done')
    return reconstructed_signals


def resampling(undersampled_data_list,target_length):
    """
    Resample each time series in the list to match the length of the longest time series.

    Parameters:
    undersampled_data_list (list): List of numpy arrays, where each array is a time series data.

    Returns:
    list: List of resampled data arrays.
    """
    # Find the length of the longest time series

    def recovering(data, target_length):
        """
        Recover the time series data to match a target length using interpolation.

        Parameters:
        data (numpy.ndarray): Original time series data.
        target_length (int): The target length to match.

        Returns:
        numpy.ndarray: The resampled time series data.
        """
        # If current length is greater than or equal to the target, return original data
        if len(data) >= target_length:
            return data

        # Interpolation
        original_indices = np.linspace(0, len(data)-1, num=len(data))
        target_indices = np.linspace(0, len(data)-1, num=target_length)
        interpolator = interp1d(original_indices, data, kind='linear')
        resampled_data = interpolator(target_indices)

        return resampled_data

    # Resample each time series in the list
    resampled_data_list = [recovering(data,target_length) for data in undersampled_data_list]

    return resampled_data_list
