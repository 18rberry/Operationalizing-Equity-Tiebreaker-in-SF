import sys
import pandas as pd
sys.path.append('../..')

from src.d01_data.block_data_api import BlockDataApi

columns_selected = ['2010 total population count',
                    "AALPI all TK5 stu 2017",
                    "ACS 2013-17 est median HH income",
                    "ACS 2013-17 est% HH below poverty lvl",
                    'ACS 2013-17 est% aged5+ Engl "not well"',
                    "SFHA_ex_Sr",
                    "num of SBAC L1 scores 4-9 2015-18"]

##1. Functions for preprocessing demographic block data:

def get_empty_cols(block_df, empty_str=["--"]):
    empty_cols = []
    for col in block_df.columns:
        for string in empty_str:
            if set(block_df[col].values) == {string}:
                empty_cols.append(col)
    return col


def clean_block_data(block_df, columns=columns_selected):
    
    clean_df = block_df.copy()[columns]
    
    #1. Turn columns which have no meaningful string value into NaN columns:
    remove_cols = get_empty_cols(clean_df)
    clean_df[remove_cols] = float("nan")
    
    #2. Turn the "--" values into zero:
    clean_df = clean_df.replace({"--": 0.})
    
    #3. Convert yes/no into binary variables:
    clean_df = clean_df.replace({"no": 0, "yes": 1})
    
    return clean_df

##2. Functions for preprocessing FRL/AALPI block data:

def frl_raw_to_pc(frl_df, both_from_count=True):
    
    #1. Initialize the new Dataframe:
    new_df = pd.DataFrame()
    
    #2. Clean the Geoid10 to match the map:
    new_df['Geoid10'] = frl_df['Geoid10'].apply(lambda x: '0%i' % int(x))
    
    #3. Load the columns with their simplified names:
    new_df["COUNT"] = frl_df["4YR AVG Student Count"]
    new_df["FRL"] = frl_df["4YR AVG FRL Count"]
    new_df["AALPI"] = frl_df["4YR AVG Eth Flag Count"]
    new_df["BOTH"] = frl_df["4YR AVG Combo Flag Count"]
    
    #4. Add relative columns:
    new_df["AALPI (%)"] = new_df["AALPI"]/new_df["COUNT"]
    new_df["FRL (%)"] = new_df["FRL"]/new_df["COUNT"]
    
    #Two possible normalizations for the intersection!
    if both_from_count:
        new_df["BOTH (%)"] = new_df["AALPI"]/new_df["COUNT"]
    else:
        new_df["BOTH (%)"] = new_df["BOTH"]/(new_df["AALPI"] + new_df["FRL"] - new_df["BOTH"])
    
    return new_df