import numpy as np
import pandas as pd
import scipy.stats
import math
def process( a , b  , sigma, deltaT,T ):
    # 2*a*b >> sigma**2
    # a,b >0
    N= round( math.floor(T/deltaT) )
    longterm_stddev = np.sqrt( b*sigma**2/(2*a) )

    process= np.zeros(N+1)
    process[0]= scipy.stats.truncnorm.rvs(-1., 1. )*longterm_stddev+b

    for i in range(N):
        stddev = sigma*np.sqrt( np.amax( [process[i]*deltaT , 0 ]))
        deltaW= np.random.normal(loc= 0, scale= 1) *stddev
        process[i+1]= process[i]+a*( b- process[i] )*deltaT +deltaW

    return process-b

def process_generator( **dictionary):
    while True :
        yield process( **dictionary ) #1 dimension process: [values]
