import numpy as np

def get_group_value(x):
    return x.iloc[0]

def get_label(x, solution_set):
    if np.isfinite(x):
        return 1. if x in solution_set else 0.
    else:
        return np.nan