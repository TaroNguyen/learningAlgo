import numpy as np
import pandas as pd
import scipy.stats
import math
# checked with pandas, this method generates no null value
def process( mu , deltaT,T ):

    N= round( math.floor(T/deltaT) )

    process= np.ones(N+1)*mu #length

    return process

def process_generator( **dictionary):
    while True :
        yield process( **dictionary ) #1 dimension process: [values]
