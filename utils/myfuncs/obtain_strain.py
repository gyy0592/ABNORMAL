

import numpy as np
from gwpy.timeseries import TimeSeries
from .extract_time import extract_time
import os 



def get_strain_data(detector, gps_start, gps_end, data_dir='/home/share/GW_data/data/LIGO/'):
    # obtain all the files under the data_dir
    fileNames = [file for file in os.listdir(data_dir) if file.endswith('.hdf5')]
    if detector =='L1':
        # select the files contain L-L1
        fileNames = [file for file in fileNames if 'L-L1' in file]
    elif detector == 'H1':
        # select the files contain H-H1
        fileNames = [file for file in fileNames if 'H-H1' in file]
    gps_start_files = []
    gps_end_files = []
    for fileName in fileNames:
        gps_start_file = extract_time(fileName)
        gps_end_file = gps_start_file + 4096
        gps_start_files.append(gps_start_file)
        gps_end_files.append(gps_end_file)
    # sort the gps_start_files and gps_end_files and fileNames
    gps_start_files, gps_end_files, fileNames = zip(*sorted(zip(gps_start_files, gps_end_files, fileNames)))

    # find the closest gps_start_file that is less than gps_start and obtain the index
    closest_gps_start_idx = max([i for i, x in enumerate(gps_start_files) if x <= gps_start])
    # find the closest gps_end_file that is larger than gps_end and obtain the index
    closest_gps_end_idx = min([i for i, x in enumerate(gps_end_files) if x >= gps_end ])
    t0_start = TimeSeries.read(os.path.join(data_dir, fileNames[closest_gps_start_idx]), format='hdf5.gwosc').t0.value
    t0_end = TimeSeries.read(os.path.join(data_dir, fileNames[closest_gps_end_idx]), format='hdf5.gwosc').t0.value
    if t0_end-t0_start > 4096 + (gps_end - gps_start):
        print(f"Data is not continuous")
        raise ValueError("Data is not continuous, missing files")
    # check the continuity of the gps_files to make sure all the data is in there
    num_files = closest_gps_end_idx - closest_gps_start_idx + 1
    # if num_files != int(np.ceil((gps_end - gps_start)/4096)):
    if num_files < int(np.ceil((gps_end - gps_start)/4096)):
        print(f'debugging log: duration = {gps_end - gps_start}, num_files = {num_files}, ceil = {np.ceil((gps_end - gps_start)/4096)}')

        raise ValueError("Data is not continuous, missing files")
    # select files contains the strain data
    selected_files = fileNames[closest_gps_start_idx:closest_gps_end_idx + 1]
    # urls = get_urls(detector, gps_start, gps_end)
    timeseries_values = []  # This will hold numpy arrays of the time series data
    for file_name in selected_files:
        # if not os.path.exists(data_dir + file_name):
        #     straindata = requests.get(url).content
        #     with open(data_dir + file_name, 'wb') as strainfile:
        #         strainfile.write(straindata)
        # legacy
        # file_path = data_dir + file_name
        # use a better way to join path
        file_path = os.path.join(data_dir, file_name)
        strain = TimeSeries.read(file_path, format='hdf5.gwosc')
        crop_start = max(gps_start, strain.t0.value)
        crop_end = min(gps_end, strain.t0.value + strain.duration.value)
        if crop_start < crop_end:
            cropped_strain = strain.crop(crop_start, crop_end)
            timeseries_values.append(cropped_strain.value)
    if timeseries_values:
        combined_strain_values = np.concatenate(timeseries_values)
        return combined_strain_values
    else:
        return np.array([])
    
def split_strain_into_segments(strain_values, seg_len, num_overlap = 1003, fs=4096):
     
    step = seg_len - num_overlap * 2   # Adjust step to ensure coverage considering the transient lengths
    num_segments = int(np.ceil((len(strain_values) - 2*num_overlap) / step))
    segments = []
    time_vectors = []

    for i in range(num_segments):
        if i == 0:
            start_index = 0
            end_index = seg_len-num_overlap 
            # 0 - 11, 9-21, 19-31
            #  1-10 10-20  20-30
            #  1-11 9-21
            #  0-10 10-20
            # we have t0 , 0-10 -> t0-t0+10
            # now we are working on t0 = t0'+ numoverlap
            
        else:
            start_index = i * step - num_overlap
            end_index = start_index + seg_len
        # Adjust indices to include transient length cut off at the beginning and end of each segment
        adjusted_start_index = start_index + num_overlap
        adjusted_end_index = end_index - num_overlap
        if end_index > len(strain_values):
            end_index = len(strain_values)
            adjusted_end_index = len(strain_values) - num_overlap

        # print('start_index', start_index, 'end_index', end_index)
        segments.append(strain_values[start_index:end_index])
        # Calculate time for each segment considering the adjusted indices
        start_time = adjusted_start_index / fs
        end_time = adjusted_end_index / fs
        time_vectors.append([start_time, end_time])
    time_vectors = np.array(time_vectors)
    return segments, time_vectors

def identify_operation_run(start_time, end_time):
  """
  This function identifies which operation run a given time range belongs to.

  Args:
      start_time: An integer representing the start GPS time.
      end_time: An integer representing the end GPS time.

  Returns:
      A string indicating the operation run the time range belongs to, 
      or "Not in any operation run" if it doesn't belong to any.
  """
  operation_runs = {
      "O1": (1126051217, 1137254417),
      "O2": (1164556817, 1187733618),
      "O3a": (1238112018, 1253977218),
      "O3b": (1256657218, 1269363618),
  }

  for operation, (known_start, known_end) in operation_runs.items():
    # Check if the provided start and end time completely falls within an operation run
    if known_start <= start_time and end_time <= known_end:
      return f"{operation}"
    # Additionally check if the provided time range overlaps with the operation run
    elif (known_start <= start_time and start_time <= known_end) or (known_start <= end_time and end_time <= known_end):
      return f"Segment overlaps with Operation Run {operation}"

  return "Not in any operation run"