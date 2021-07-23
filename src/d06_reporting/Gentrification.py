import pandas as pd
import numpy as np
#import geopandas as gpd
import matplotlib.pyplot as plt
import collections
import seaborn as sns

try:
    collectionsAbc = collections.abc
except AttributeError:
    collectionsAbc = collections
    
from collections import OrderedDict
#from collections.abc import Iterable
import matplotlib.pyplot as plt

import sys
sys.path.append('../')

class Gentrification: 
    def gentrification_data():
        df = pd.read_csv("~/displacement-typologies/data/outputs/typologies/SanFrancisco_typology_output.csv", sep=",")
        block_df_dict = pd.read_excel("/share/data/school_choice/Data/SF 2010 blks 022119 with field descriptions (1).xlsx", None)
        block_database = block_df_dict["block database"]
        block_database = block_database.dropna(axis=0, how='all')
        block_database = block_database.dropna(axis=1, how='all')
        new_blockdf = block_database[["Tract", "CTIP_2013 assignment"]]
        ctip_data = new_blockdf.merge(df, left_on = "Tract", right_on = "GEOID", how = "left")
        ctip_data = ctip_data[["Tract", "CTIP_2013 assignment", "OD", "ARG", "EOG", "AdvG", "SMMI", "ARE", "BE", "SAE"]]
        ctip_data = ctip_data.dropna(axis=0, how='all')
        ctip_data = ctip_data.dropna(axis=1, how='all')
        ctip_data_new = ctip_data.drop_duplicates()
        ctip_data_new = ctip_data_new.groupby("CTIP_2013 assignment").sum()
        ctip_data_removed = ctip_data_new.drop("Tract", axis=1)
        print(sns.heatmap(ctip_data_removed))
        
    gentrification_data()