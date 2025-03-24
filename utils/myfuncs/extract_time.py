def extract_time(filename):
    # Split the filename by '-'
    parts = filename.split('-')

    # The time info is the second last part of the split filename
    time_info = parts[-2]

    return float(time_info)