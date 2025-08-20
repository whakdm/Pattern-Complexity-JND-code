import numpy as np

def func_randnum(col, row):
    mask = np.random.rand(col, row)
    randmat = np.where(mask >= 0.5, 1, -1)
    return randmat