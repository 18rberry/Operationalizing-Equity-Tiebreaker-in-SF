import numpy as np

def get_group_value(x):
    return x.iloc[0]

#Function that transforms index set in binary array for classifier:
def get_label(x, solution_set, block_idx=None):
    if block_idx is None:
        if np.isfinite(x):
            return 1. if x in solution_set else 0.
        else:
            return np.nan
    else:
        if np.isfinite(x) and x in block_idx:
            return 1. if x in solution_set else 0.
        else:
            return np.nan

#Function that adds percent columns to the FRL data
# RMK: column names as in classifier data api
def add_percent_columns(frl_df):
    
    frl_df['pctFRL'] = frl_df['nFRL'] / frl_df['n']
    frl_df['pctAALPI'] = frl_df['nAALPI'] / frl_df['n']
    frl_df['pctFocal'] = frl_df['nFocal'] / frl_df['n']
    
    frl_df['pctBoth'] = frl_df['nBoth'] / frl_df['n']
    frl_df['pctBothUnion'] = frl_df['nBoth'] / frl_df['nFocal'] #union normalization
    
    return frl_df

#Function that adds group criteirion columns to the dataframe:
def add_group_columns(group_type, len_BG=8):
    return None