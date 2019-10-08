import numpy as np
import scipy.linalg as scp

def hat(vector:np.ndarray):
    ret = np.zeros((3,3))
    ret[0,1] = -vector[2]
    ret[0,2] = vector[1]
    ret[1,0] = vector[2]
    ret[1,2] = -vector[0]
    ret[2,0] = -vector[1]
    ret[2,1] = vector[0]
    return ret

def expm(vector: np.ndarray) -> np.ndarray:
    return scp.expm(vector)