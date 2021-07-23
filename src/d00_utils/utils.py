import numpy as np
import matplotlib.pyplot as plt

def get_group_value(x):
    return x.iloc[0]

#Function that transforms index set in binary array for classifier:
def get_label(x, solution_set):
    if np.isfinite(x):
        return 1. if x in solution_set else 0.
    else:
        return np.nan

#Function to plot a column of a geodataframe:
def plot_map_column(map_df_data, col, cmap="viridis", ax=None, save=False, leg=True):
        
    if ax is None:
        fig, ax = plt.subplots(figsize=(30,30))
        if save:
            save = True
    elif save:
        fig = plt.gcf()

    map_df_data.plot(column=col, ax=ax, cmap=cmap,
                     legend=leg, legend_kwds={'orientation': "horizontal"},
                     missing_kwds={'color': 'lightgrey'})
    
    ax.set_title(col, fontsize=50)
    plt.tight_layout()
    plt.show()
        
    if save:
        fname = col + '.png'
        fig.savefig(fname)

    return ax

#Function that adds percent columns to the FRL data
# RMK: column names as in classifier data api
def add_percent_columns(frl_df):
    
    frl_df['pctFRL'] = frl_df['nFRL'] / frl_df['n']
    frl_df['pctAALPI'] = frl_df['nAALPI'] / frl_df['n']
    frl_df['pctFocal'] = frl_df['nFocal'] / frl_df['n']
    
    frl_df['pctBoth'] = frl_df['nBoth'] / frl_df['n']
    frl_df['pctBothUnion'] = frl_df['nBoth'] / frl_df['nFocal'] #union normalization
    
    return frl_df

#Function that adds block-group and neighborhood aggregated columns:
def add_BG_columns(df, positive_group="nFocal", len_BG=8):
    
    df["BlockGroup"] = df["BlockGroup"].astype(str).str.slice(stop=len_BG+1)
    
    agg_df = df[["BlockGroup", "n", positive_group]].groupby(['BlockGroup']).sum().reset_index()
    agg_df = agg_df.rename(columns={"n": "BG_n", positive_group: "BG_" + positive_group})
    
    extended_df = df.merge(agg_df, on="BlockGroup")
    extended_df["BG_nOther"] = extended_df['BG_n'] - extended_df["BG_" + positive_group]
    extended_df["BG_pct" + positive_group[1:]] = extended_df["BG_" + positive_group]/extended_df['BG_n']
    
    return extended_df