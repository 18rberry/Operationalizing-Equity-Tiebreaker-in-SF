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

    def __init__(self):
        super().__init__()
        self.gentrification_cols = ["OD", "ARG", "EOG", "AdvG", "SMMI", "ARE", "BE", "SAE"]
        
    def gentrification_data(self, df, block_df_dict):
        block_database = block_df_dict["block database"].dropna(axis=0, how='all').dropna(axis=1, how='all')
        new_blockdf = block_database[["Tract", "CTIP_2013 assignment"]]
        ctip_data = new_blockdf.merge(df, left_on = "Tract", right_on = "GEOID", how = "left")
        ctip_data = ctip_data[["Tract", "CTIP_2013 assignment", "OD", "ARG", "EOG", "AdvG", "SMMI", "ARE", "BE", "SAE"]]
        ctip_data = ctip_data.dropna(axis=0, how='all').dropna(axis=1, how='all').drop_duplicates()
        ctip_data_new = ctip_data.groupby("CTIP_2013 assignment").sum()
        ctip_data_removed = ctip_data_new.drop("Tract", axis=1)
        return sns.heatmap(ctip_data_removed)
     
    def frl_vs_ctip(self, updated_FRL, block_df_dict): 
        block_database = block_df_dict["block database"].dropna(axis=0, how='all').dropna(axis=1, how='all')
        grouped_Geoid = updated_FRL["Grouped GeoID External"]
        grouped_Geoid_filtered = grouped_Geoid[grouped_Geoid["Geoid Group"].astype('str').str.len() > 3]
        new_merge = grouped_Geoid_filtered.merge(block_database, left_on = "Geoid Group", right_on = "Block")
        new_merge_grouped = new_merge.groupby("CTIP_2013 assignment").mean()
        new_merge_grouped = new_merge_grouped[["4YR AVG Student Count", "4YR AVG FRL Count",
                                      "4YR AVG Eth Flag Count", "4YR AVG Combo Flag Count"]]
        new_merge_grouped["FRL"] = (new_merge_grouped["4YR AVG FRL Count"] - new_merge_grouped["4YR AVG Combo Flag Count"])/new_merge_grouped["4YR AVG Student Count"]
        new_merge_grouped["AALPI"] = (new_merge_grouped["4YR AVG Eth Flag Count"] - new_merge_grouped["4YR AVG Combo Flag Count"])/new_merge_grouped["4YR AVG Student Count"]
        new_merge_grouped["Combo"] = new_merge_grouped["4YR AVG Combo Flag Count"]/new_merge_grouped["4YR AVG Student Count"]
        new_merge_grouped["Other"] = (new_merge_grouped["4YR AVG Student Count"] - ((new_merge_grouped["4YR AVG FRL Count"] + new_merge_grouped["4YR AVG Eth Flag Count"]) - new_merge_grouped["4YR AVG Combo Flag Count"]))/new_merge_grouped["4YR AVG Student Count"]
        ax = sns.heatmap(new_merge_grouped[["Combo", "FRL", "AALPI", "Other"]], cmap = "YlOrRd")
        plt.title("4YR Average of Student Demographic Counts", fontsize = 15)
        return ax 
    
    def gentrification_vs_demo(self, updated_FRL, df):
        grouped_Geoid = updated_FRL["Grouped GeoID External"]
        grouped_Geoid_filtered = grouped_Geoid[grouped_Geoid["Geoid Group"].astype('str').str.len() > 3]
        grouped_Geoid_filtered["New Geoid"] = grouped_Geoid_filtered["Geoid Group"].astype(str).str[:10].astype(int)
        grouped_Geoid_filtered = grouped_Geoid_filtered.merge(df, left_on = "New Geoid", right_on = "GEOID")
        grouped_Geoid_new = grouped_Geoid_filtered[["GEOID", "4YR AVG Student Count", "4YR AVG FRL Count", 
                                                    "4YR AVG Eth Flag Count", "4YR AVG Combo Flag Count",
                                            "OD", "ARG", "EOG", "AdvG", "SMMI", "ARE", "BE", "SAE"]]
        grouped_Geoid_new = grouped_Geoid_new.drop_duplicates().dropna(axis=0, how='all').dropna(axis=1, how='all')
        grouped_Geoid_new = grouped_Geoid_new.groupby("GEOID").sum()
        #gentrification_cols = ["OD", "ARG", "EOG", "AdvG", "SMMI", "ARE", "BE", "SAE"]
        gentrification_values = []

        for i in grouped_Geoid_new.index: 
            new_value = None
            for col in self.gentrification_cols: 
                if grouped_Geoid_new.loc[i, col] != 0.0: 
                    new_value = col
            gentrification_values.append(new_value)
            
        grouped_Geoid_new["Gentrification"] = gentrification_values
        grouped_Geoid_gent = grouped_Geoid_new[grouped_Geoid_new["Gentrification"].notnull()]
        grouped_Geoid_gent = grouped_Geoid_gent[["4YR AVG Student Count", "4YR AVG FRL Count", "4YR AVG Eth Flag Count", "4YR AVG Combo Flag Count", "Gentrification"]]
        return grouped_Geoid_gent
    
    def grouped_barchart(self, updated_FRL, df):
        grouped_Geoid_gent = self.gentrification_vs_demo(updated_FRL, df)
        gentrification_agg = grouped_Geoid_gent.groupby("Gentrification").mean()
        gentrification_agg_new = gentrification_agg.drop("4YR AVG Student Count", axis = 1)
        return [gentrification_agg_new.loc[self.gentrification_cols].plot.bar(), gentrification_agg]
        
    def stacked_barchart(self, updated_FRL, df):
        labels = ["Student", "FRL", "AALPI", "Combo"]
        gentrification_agg = self.grouped_barchart(updated_FRL, df)[1]
        plt.bar(self.gentrification_cols, gentrification_agg["4YR AVG Student Count"])
        plt.bar(self.gentrification_cols, gentrification_agg["4YR AVG FRL Count"])
        plt.bar(self.gentrification_cols, gentrification_agg["4YR AVG Eth Flag Count"])
        plt.bar(self.gentrification_cols, gentrification_agg["4YR AVG Combo Flag Count"])
        plt.legend(labels)
    
    

