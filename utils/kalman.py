import numpy as np
import pandas as pd
def kalman_coeff(sigmaS ,lbd, sigmamu,deltaT):


    longterm_stdvar = sigmamu**2/(2*lbd) #true

    f = ( sigmaS**2/deltaT + longterm_stdvar )*( 1- np.exp(-2*lbd*deltaT)) #true

    g_square = f**2 + 2*(sigmaS**2)*(sigmamu**2)/( lbd*deltaT) \
        * ( np.exp(- 2*lbd*deltaT) - np.exp(- 4*lbd*deltaT) ) #true
    g = np.sqrt(g_square) #true

    gamma = (g-f)/( 2* np.exp(- 2*lbd*deltaT) ) #true

    k = gamma/(gamma+ sigmaS**2/deltaT)

    return k, (1-k)*np.exp(- lbd*deltaT)
def kalman_estimate( array, sigmaS ,lbd, sigmamu,deltaT):
    k, one_minus_alpha = kalman_coeff(sigmaS ,lbd, sigmamu,deltaT)
    data = pd.DataFrame()
    data['last returns'] = array
    data['kalman'] = k/(1-one_minus_alpha)*data['last returns'].ewm( alpha = 1-one_minus_alpha, adjust=False).mean()
    return data['kalman'].values
