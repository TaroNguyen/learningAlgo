import numpy as np
import pandas as pd
import scipy.stats
import math

def process( lbd, sigma, deltaT,T ):

    N= round( math.floor(T/deltaT) )
    invariant_std= sigma/np.sqrt(2*lbd)

    process= np.zeros(N+1) #length
    process[0]= scipy.stats.truncnorm.rvs(-1.96, 1.96 )

    for i in range(N):
        deltaW= np.random.normal(loc= 0, scale= np.sqrt(1-np.exp(-2*lbd*deltaT)))
        process[i+1]= np.exp(-lbd*deltaT)*process[i]+deltaW
    return invariant_std*process

def process_generator( **dictionary):
    while True :
        yield process( **dictionary ) #1 dimension process: [values]
