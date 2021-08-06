import sys
import pandas as pd
sys.path.append('../..')


columns_selected = ['2010 total population count',
                    "AALPI all TK5 stu 2017",
                    "ACS 2013-17 est median HH income",
                    "ACS 2013-17 est% HH below poverty lvl",
                    'ACS 2013-17 est% aged5+ Engl "not well"',
                    "SFHA_ex_Sr",
                    "num of SBAC L1 scores 4-9 2015-18"]


def get_empty_cols(block_df: pd.DataFrame, empty_str=None):
    """
    Get columns that are empty
    :param block_df: pandas.DataFrame with block data
    :param empty_str:
    :return:
    """
    if empty_str is None:
        empty_str = ["--"]
    empty_cols = []
    for col in block_df.columns:
        for string in empty_str:
            if set(block_df[col].values) == {string}:
                empty_cols.append(col)
    return empty_cols


def clean_block_data(block_df, columns=None):
    """
    Clean block data:
        1. Turn columns which have no meaningful string value into NaN columns
        2. Turn the "--" values into zero
        3. Convert yes/no into binary variables
    :param block_df: pandas.DataFrame with block data
    :param columns:
    :return:
    """
    if columns is None:
        columns = columns_selected
    
    clean_df = block_df.copy()[columns]

    remove_cols = get_empty_cols(clean_df)
    clean_df[remove_cols] = float("nan")
    clean_df = clean_df.replace({"--": 0.})
    clean_df = clean_df.replace({"no": 0, "yes": 1})
    
    return clean_df
