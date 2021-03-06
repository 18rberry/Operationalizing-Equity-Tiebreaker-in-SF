import numpy as np
import pandas as pd


def get_group_value(x: pd.Series):
    """
    Get single value from pandas.Series with constant values.
    :param x: pandas.Series
    :return: single value
    """
    return x.iloc[0]


def get_label(x, solution_set, block_idx=None):
    """
    Function that transforms index set in binary array for classifier
    :param x:
    :param solution_set: array with the solution set
    :param block_idx: array with the block index (work around in special cases)
    :return:
    """
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


def add_percent_columns(frl_df: pd.DataFrame):
    """
    Function that adds percent columns to the FRL data
    RMK: column names as in classifier data api
    :param frl_df: pandas.DataFrame with the focal student data
    :return:
    """
    
    frl_df['pctFRL'] = frl_df['nFRL'] / frl_df['n']
    frl_df['pctAALPI'] = frl_df['nAALPI'] / frl_df['n']
    frl_df['pctFocal'] = frl_df['nFocal'] / frl_df['n']
    
    frl_df['pctBoth'] = frl_df['nBoth'] / frl_df['n']
    frl_df['pctBothUnion'] = frl_df['nBoth'] / frl_df['nFocal'] #union normalization
    if 'nAAFRL' in frl_df.columns:
        frl_df['pctAAFRL'] = frl_df['nAAFRL'] / frl_df['n']
    
    return frl_df


def add_group_columns(df: pd.DataFrame, group_type, len_BG=8, positive_group="nFocal"):
    """
    Function that adds group criteirion columns to the dataframe
    :param df: pandas.DataFrame
    :param group_type: type of grouping
    :param len_BG: length of block group code
    :param positive_group: column name of positive group
    :return:
    """
    # Select the bg_col based on the group_type. These columns must have been added to the df:
    if group_type == "block_group":
        bg_col = "BlockGroup"
        df["BlockGroup"] = df.index.to_series().astype(str).str.slice(stop=len_BG+1)
    elif group_type == "nbhd":
        bg_col = "Neighborhood"
    else:
        print("Grouping method must be nbhd or block_group. No grouping defined.")
        return df

    # Create an aggregated dataframe with counts per block group:
    agg_df = df[[bg_col, "n", "nFocal", "nFRL", "nAALPI", "nBoth"]].groupby(bg_col).sum()
    agg_df = agg_df.rename(columns={"n": "BG_n",
                                    "nFocal": "BG_nFocal",
                                    "nFRL": "BG_nFRL",
                                    "nAALPI": "BG_nAALPI",
                                    "nBoth": "BG_nBoth"})
    # Merge the aggregated df on the main df via the bg_column:
    df["geoid"] = df.index
    extended_df = df.merge(agg_df, on=bg_col).set_index("geoid")

    extended_df["BG_nOther"] = extended_df['BG_n'] - extended_df["BG_" + positive_group]

    extended_df['BG_pctFRL'] = extended_df['BG_nFRL'] /  extended_df['BG_n']
    extended_df['BG_pctAALPI'] = extended_df['BG_nAALPI'] / extended_df['BG_n']
    extended_df['BG_pctFocal'] = extended_df['BG_nFocal'] /  extended_df['BG_n']

    extended_df['BG_pctBoth'] = extended_df['BG_nBoth'] / extended_df['BG_n']
    extended_df['BG_pctBothUnion'] = extended_df['BG_nBoth'] / extended_df['BG_nFocal'] #union

    return extended_df


def add_bayesian_bernoulli(frl_df):
    """
    Compute focal students posterior for uniform priors
    :param frl_df: pandas.DataFrame with the focal student data
    :return:
    """
    variables = ['FRL', 'AALPI', 'Both', 'Focal']
    
    for x in variables:
        prob_key = 'prob' + x
        n_key = 'n' + x
        frl_df[prob_key] = (frl_df[n_key] + 1) / (frl_df['n'] + 2)
        
    return frl_df


def show_values_on_bars(ax, h_v="v", space=0.4):
    """
    Add bar values to barplot
    :param ax: (optional) axis values
    :param h_v: barplot orientation ('v' or 'h')
    :param space: space size
    :return:
    """
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                if p.get_height() * 100. < 1:
                    continue
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() / 2
                value = "%.1f%%" % (p.get_height() * 100.)
                ax.text(_x, _y, value, ha="center") 
        elif h_v == "h":
            for p in ax.patches:
                if p.get_width() * 100. < 1:
                    continue
                _x = p.get_x() + p.get_width()  # + float(space)
                _y = p.get_y() + p.get_height()
                value = "%.1f%%" % (p.get_width() * 100.)
                ax.text(_x, _y, value, ha="left")

    if isinstance(ax, np.ndarray):
        for idx, ax in np.ndenumerate(ax):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(ax)