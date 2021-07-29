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
def add_group_columns(df, group_type, len_BG=8, positive_group="nFocal"):
    #Select the bg_col based on the group_type. These columns must have been added to the df:
    if group_type == "block_group":
        bg_col = "BlockGroup"
        df["BlockGroup"] = df.index.to_series().astype(str).str.slice(stop=len_BG+1)
    elif group_type == "nbhd":
        bg_col = "Neighborhood"
    else:
        print("Grouping method must be nbhd or block_group. No grouping defined.")
        return df
    
    #Create an aggregated dataframe with counts per block group:
    agg_df = df[[bg_col, "n", "nFocal", "nFRL", "nAALPI", "nBoth"]].groupby(bg_col).sum()
    agg_df = agg_df.rename(columns={"n": "BG_n",
                                    "nFocal": "BG_nFocal",
                                    "nFRL": "BG_nFRL",
                                    "nAALPI": "BG_nAALPI",
                                    "nBoth": "BG_nBoth"})
    #Merge the aggregated df on the main df via the bg_column:
    df["geoid"] = df.index
    extended_df = df.merge(agg_df, on=bg_col).set_index("geoid")
    
    extended_df["BG_nOther"] = extended_df['BG_n'] - extended_df["BG_" + positive_group]
    
    extended_df['BG_pctFRL'] = extended_df['BG_nFRL'] /  extended_df['BG_n']
    extended_df['BG_pctAALPI'] = extended_df['BG_nAALPI'] / extended_df['BG_n']
    extended_df['BG_pctFocal'] = extended_df['BG_nFocal'] /  extended_df['BG_n']
    
    extended_df['BG_pctBoth'] = extended_df['BG_nBoth'] / extended_df['BG_n']
    extended_df['BG_pctBothUnion'] = extended_df['BG_nBoth'] / extended_df['BG_nFocal'] #union
    
    return extended_df