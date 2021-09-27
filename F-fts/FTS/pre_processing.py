
import numpy as np

def norm(x):
    ''' Min-Max Feature scaling '''
    return (x - min(x)) / (max(x) - min(x))

def standardization(x):
    '''padronização dos dados'''
    return x - np.mean(x) /  np.std(x)  