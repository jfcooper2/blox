# utils.py
# A few utility functions for defining workloads and jobs

import math
import random
import numpy as np

# In iterations per second
def get_random_gpu_tputs():
    def rand_int():
        multiplier = 1
        t = 1
        r = random.random()
        # if r < 0.8: t = 1
        # elif r < 0.95: t = 0.5
        # else: t = 0.2
        return t * multiplier

    tputs = {}
    tputs['K80'] = 1 * rand_int()
    tputs['V100'] = 4 * rand_int()
    tputs['P100'] = 9 * rand_int()
    return tputs

def poisson_next_arrival_time(jobs_per_hour):
    """
    Return the time before the next job arrives, in seconds
    """
    next_arrival_in_hours = -math.log(1.0 - random.random()) / jobs_per_hour 
    next_arrival_in_seconds = math.ceil(next_arrival_in_hours * 60 * 60)
    return next_arrival_in_seconds

def exponential(lambd):
    """
    An iterator for exponentially distributed data
    """
    while True:
        yield random.expovariate(lambd)


def get_total_iteration(range_min, range_max):
    """
    Return a random total between range_min and range_max
    """
    return random.randint(range_min, range_max)

def get_job_gpu_demand():
    """
    Return a random value from {1,2,4,8} for this gpu 
    """
    rand_var = random.uniform(0,1)
    if rand_var >= 0.95:
        return 8
    elif 0.8 <= rand_var < 0.95:
        return 4
    elif 0.7 <= rand_var < 0.8:
        return 2
    else:
        return 1
    
def get_total_iteration_exp(range_min, range_max):
    """
    Returns an exponentially distributed sample from within the 
    provided range [range_min, range_max]
    """
    x = 0
    while not range_min <= x <= range_max:
        x = math.ceil(random.expovariate(1/60000))
    return x


def get_gavel_like_iter():
    """
    Return a sample from a power-law mixture model
    """
    if random.random() >= 0.8:
        iters = 60 * (10 ** random.uniform(3,4))
    else:
        iters = 60 * (10 ** random.uniform(1.5,3))
    return iters

def small_trace_dur():
    """
    Return a sample from a power-law mixture model
    """
    if random.random() >= 0.8:
        iters = 60 * (10 ** random.uniform(2.5,2.7))
    else:
        iters = 60 * (10 ** random.uniform(1.5,2.5))
    return iters
        

def gpu_normalized_vector(vector):
    """
    Return scaled version of vector, where result[0] = 1
    """
    return [item/vector[0] for item in vector]

def cumulative_map(orig_map, new_map):
    """
    Adds the entries of new map into orig and returns orig
    """
    for key, value in new_map.items():
        if key in orig_map:
            orig_map[key] += value
        else:
            orig_map[key] = value

    return orig_map
