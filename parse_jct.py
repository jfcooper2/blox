import json
import sys
import statistics
import numpy as np

def main(file_name):
    """
    Get avg jct
    """
    with open(file_name, "r") as fin:
        data_job = json.load(fin)
    avg_jct = _get_avg_jct(data_job)
    print("Avg JCT {}".format(avg_jct))
    # print(f'Median JCT {_get_median_jct(data_job)}')

    values = list(data_job.values())
    completion_times = [v[1] - v[0] for v in values]
    percentiles  = [ 5, 50, 95 ]
    for p in percentiles:
        jct_perc = np.percentile(completion_times, p)
        print(f'{p} percentile JCT: {jct_perc}')



def _get_avg_jct(time_dict):
    """
    Fetch the avg jct from the dict
    """
    values = list(time_dict.values())
    count = 0
    jct_time = 0
    for v in values:
        jct_time += v[1] - v[0]
        count += 1

    return jct_time / count

def _get_median_jct(time_dict):
    values = list(time_dict.values())
    completion_times = [v[1] - v[0] for v in values]
    return np.percentile(completion_times, 50)
    # return statistics.median(completion_times)

if __name__ == "__main__":
    main(sys.argv[1])
