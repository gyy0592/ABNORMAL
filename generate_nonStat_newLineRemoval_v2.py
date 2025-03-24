

import numpy as np
import os 
from utils.myfuncs.obtain_strain import get_strain_data, split_strain_into_segments, identify_operation_run
from utils.myfuncs.data_conditioning import dumpNotchIIR_v3, obtain_smooth_asd
from gwpy.timeseries import TimeSeries
from utils.myfuncs.temp_getLossDis import obtain_lossDis
from utils.myfuncs.generate_noise import generate_noise_from_psd
import pickle
import glob 

data_dir = '/home/share/GW_data/data/LIGO/'
outputPrefix = 'threshold_newRemoval_v2/'
# create output directory if not exist
os.makedirs(outputPrefix, exist_ok=True)

if __name__ == '__main__':
    merged_ll_segments = [(1126076416.0, 1126084607.0), (1246453760.0, 1246461951.0), 
                         (1171222528.0, 1171267583.0), (1260441600.0, 1260539904.0)]
    fs = 4096
    numtaps = 1001
    operationNames = ['O1', 'O2', 'O3a', 'O3b']
    detector_names = ['L1', 'H1']
    weights_path = '/weights/weights_epoch199.pth'
    weights_path = None
    # Loop through all segments
    for lp_detector in range(len(detector_names)):
        detector_name = detector_names[lp_detector]
        for lp in range(len(merged_ll_segments)):
            operation_name = identify_operation_run(*merged_ll_segments[lp])
            print(f'working on {operation_name}')

            # Define all file paths at the beginning
            real_output_path = os.path.join(outputPrefix, f'segment_real_long_bandpassed_{detector_name}_{operation_name}.pkl')
            mock_output_path = os.path.join(outputPrefix, f'segment_mock_long_bandpassed_{detector_name}_{operation_name}.pkl')
            
            tempFile_realLoss = f'temp_real_long_newRemoval_{detector_name}_{operation_name}'
            tempFile_realLoss4Selection = f'temp_real_long_newRemoval_selection_{detector_name}_{operation_name}'
            tempFile_mockLoss = f'temp_mock_long_newRemovalV2_selected_{detector_name}_{operation_name}'
            
            line_removal_path = os.path.join(outputPrefix, f'timeSeries2det_lineRemoval_{detector_name}_{operation_name}.pkl')
            loss_dis_path = os.path.join(outputPrefix, f'loss_dis_real_long_bandpassed_{detector_name}_{operation_name}.pkl')
            mock_loss_dis_path = os.path.join(outputPrefix, f'loss_dis_mock_long_bandpassed_{detector_name}_{operation_name}.pkl')
            asd_path = os.path.join(outputPrefix, f'asd_long_{detector_name}_{operation_name}.npz')

            # Get strain data and create time series
            strain_data = get_strain_data(f'{detector_name}', *merged_ll_segments[lp], data_dir=data_dir)

            # Load lines info
            lines_npz = np.load(os.path.join('./spectral_lines', f'{detector_name}_{operation_name}_lines.npz'))
            lines_band_new, stop_factors = lines_npz['lines_band'], lines_npz['stop_factors']

            # Check for existing line removal data
            if os.path.exists(line_removal_path):
                print(f"Loading existing line removal data")
                with open(line_removal_path, 'rb') as f:
                    timeSeries2det_lineRemoval = pickle.load(f)['data']
            else:
                print("Performing line removal...")
                timeSeries2det_lineRemoval = dumpNotchIIR_v3(strain_data, lines_band_new, stop_factors)
                with open(line_removal_path, 'wb') as f:
                    pickle.dump({'data': timeSeries2det_lineRemoval}, f)

            # Process segments
            method_nOverlap = 1003
            duration_per_segment = 5
            samples_per_segment = fs * duration_per_segment + method_nOverlap*2

            print("Computing segments...")
            segment_real_long_bandpassed, time_real_long_bandpasses = split_strain_into_segments(
                timeSeries2det_lineRemoval, samples_per_segment, 
                num_overlap=method_nOverlap, fs=fs
            )
            
            with open(real_output_path, 'wb') as f:
                pickle.dump(segment_real_long_bandpassed, f)

            print("Computing loss distribution...")
            loss_dis_real_long_bandpassed = obtain_lossDis(
                segment_real_long_bandpassed,
                tempFile_path=tempFile_realLoss,
                saved_model_path = weights_path
            )
            with open(loss_dis_path, 'wb') as f:
                pickle.dump({'loss': loss_dis_real_long_bandpassed, 'time': time_real_long_bandpasses}, f)

            # Process selection segments
            duration_per_segment4selection = 1000
            samples_per_segment4selection = fs*duration_per_segment4selection + 2*method_nOverlap
            # timeSeries2det_lineRemoval = strain_data[502:-502].copy()
            segment_real_long_bandpassed4selection, _ = split_strain_into_segments(
                timeSeries2det_lineRemoval[:4096*4096*3005], 
                samples_per_segment4selection, 
                num_overlap=int(samples_per_segment4selection/4), 
                fs=fs
            )
            # throw about first one and last one in case different length cause trouble 
            segment_real_long_bandpassed4selection = segment_real_long_bandpassed4selection[1:-1]

            selection_temp_path = f'/home/guoyiyang/github_repo/OoD_nonStationary/processed_data/temp/{tempFile_realLoss4Selection}'
            if glob.glob(f'{selection_temp_path}*'):
                print(f"Loading existing selected segment")
                placeholder = [np.ones(1000) for _ in range(9999)]
                loss_dis_real_long_bandpassed_selection = obtain_lossDis(
                    placeholder, 
                    tempFile_path=tempFile_realLoss4Selection,
                    saved_model_path = weights_path
                )
            else:
                print("Selecting segment with highest loss...")
                loss_dis_real_long_bandpassed_selection = obtain_lossDis(
                    segment_real_long_bandpassed4selection, 
                    tempFile_path=tempFile_realLoss4Selection,
                    saved_model_path = weights_path
                )

            mock_temp_path = f'/home/guoyiyang/github_repo/OoD_nonStationary/processed_data/temp/{tempFile_mockLoss}'
            if glob.glob(f'{mock_temp_path}*') and os.path.exists(mock_output_path):
                print(f"Loading existing mock segment")
                placeholder = [np.ones(1000) for _ in range(9999)]
                loss_dis_mock = obtain_lossDis(placeholder, tempFile_path=tempFile_mockLoss, saved_model_path = weights_path)
            else:
                print("Generating mock segment...")
                # select the segment that has the highest loss
                segment_idx = np.argsort(loss_dis_real_long_bandpassed_selection)[0]
                selected_data = segment_real_long_bandpassed4selection[segment_idx]
                asd_long = TimeSeries(selected_data, sample_rate=fs).asd(
                    fftlength=20+2*method_nOverlap/fs
                )/np.sqrt(2)
                asd_long_smooth = obtain_smooth_asd(asd_long.value)
                
                np.savez(asd_path, 
                        asd_long=asd_long, 
                        asd_long_smooth=asd_long_smooth, 
                        allow_pickle=True)
                
                long_noises_fromSmooth, _ = generate_noise_from_psd(
                    int(1.3*4096**2),
                    asd_long_smooth**2, 
                    num_noises=1, 
                    freq_range=[0, fs/2 + 100], 
                    duration=4096+4*(method_nOverlap)/fs
                )
                
                mockNoise_bandpass = dumpNotchIIR_v3(
                    long_noises_fromSmooth[0], 
                    lines_band_new, 
                    stop_factors
                )[4096*10:-4096*10]
                
                # Generate mock segments
                segment_mock = []
                num_segments = min(len(mockNoise_bandpass) // samples_per_segment, 5000)
                for i in range(num_segments):
                    start = i * samples_per_segment
                    end = start + samples_per_segment
                    segment_mock.append(mockNoise_bandpass[start:end])
            
                loss_dis_mock = obtain_lossDis(segment_mock, tempFile_path=tempFile_mockLoss, saved_model_path = weights_path)

                with open(mock_output_path, 'wb') as f:
                    pickle.dump(segment_mock, f)

            # Save loss_dis_mock
            with open(mock_loss_dis_path, 'wb') as f:
                pickle.dump({'loss': loss_dis_mock}, f)

# Run generate threshold script
# import sys, subprocess
# subprocess.run([sys.executable, "generate_threshold.py", "--output-prefix", outputPrefix])