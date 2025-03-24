

import numpy as np 
from scipy.interpolate import interp1d



def generate_noise_from_psd(N, psd, num_noises=1,freq_range=[30,1700], sample_rate = 4096, duration = None):
    freqVec = np.linspace(0, sample_rate/2, len(psd))
    noises = np.zeros((num_noises, N))
    random_asds = []
    for i in range(num_noises):
        WGN = np.random.randn(N)
        X = np.fft.rfft(WGN) / np.sqrt(N)
        asd = np.sqrt(psd)
        uneven = N % 2
        # Simulate the white noise of rFFT
        # X = (np.random.randn(N // 2 + 1 + uneven) + 1j * np.random.randn(N // 2 + 1 + uneven))
        
        selected_indices = np.where((freqVec > freq_range[0]) & (freqVec < freq_range[1]))[0]
        
        # Interpolate selected ASD values to match the length of X
        interp_asd = interp1d(freqVec[selected_indices],asd[selected_indices], kind='linear', bounds_error=False, fill_value="extrapolate")
        newFreqVec = np.fft.rfftfreq(N+uneven, d=1.0/sample_rate)
        random_asd = interp_asd(newFreqVec)
        nonSelected_indices = np.where(~ ((newFreqVec> freq_range[0]) & (newFreqVec < freq_range[1])))[0]
        random_asd[nonSelected_indices] = 1e-30
        # random_asd[random_asd<1e-30] = 1e-30

        # Apply the random ASD to create colored noise
        # In order to keep the nSample equal to before
        Y_colored = X * random_asd
        y_colored = np.fft.irfft(Y_colored).real * np.sqrt(N*sample_rate)
        if uneven:
            y_colored = y_colored[:-1]
        
        noises[i, :] = y_colored 
        random_asds.append(random_asd)
        
    return noises, random_asds