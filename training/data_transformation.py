import numpy as np
import pandas as pd
import scipy.stats
import math

def add_noise( timeserie, noise_level = 0.01*np.sqrt(1./365) ):
    n = len(timeserie)
    for i in range(n):
        timeserie[i]+= np.random.normal(loc= 0, scale= 1)*noise_level
    return timeserie

def transform( timeserie):
    timeserie = add_noise( timeserie, noise_level =0.2*np.sqrt(1./365))
    return timeserie
