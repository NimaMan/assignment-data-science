import numpy as np 
from scipy.signal import argrelextrema
    

def get_local_optimas_patterns(x, local_maxima=True):
    # return the local minimas or local maximas of functions
    if local_maxima:
        return np.diff(argrelextrema(x, np.greater))
    else:
        return np.diff(argrelextrema(x, np.less))