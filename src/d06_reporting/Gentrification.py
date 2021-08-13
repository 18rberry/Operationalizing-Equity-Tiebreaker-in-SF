import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import collections
import seaborn as sns
import geopandas as gpd

try:
    collectionsAbc = collections.abc
except AttributeError:
    collectionsAbc = collections
    
from collections import OrderedDict
#from collections.abc import Iterable
import matplotlib.pyplot as plt

import sys
sys.path.append('../')

from src.d00_utils.file_paths import SF_GENT_PATH, REDLINING_PATH


class Gentrification: 
    
    SF_gent = pd.read_csv(SF_GENT_PATH)
    
    def __init__(self):
        super().__init__()
        self.gentrification_cols = ["OD", "ARG", "EOG", "AdvG", "SMMI", "ARE", "BE", "SAE"]
        self.merged_map_tiebreaker = None
        self.params = None 
        self.missing_vals = None
        
    def gentrification_data(self, df, block_df_dict):
        block_database = block_df_dict.dropna(axis=0, how='all').dropna(axis=1, how='all')
        new_blockdf = block_database[["Tract", "CTIP_2013 assignment"]]
        ctip_data = new_blockdf.merge(df, left_on = "Tract", right_on = "GEOID", how = "left")
        ctip_data = ctip_data[["Tract", "CTIP_2013 assignment", "OD", "ARG", "EOG", "AdvG", "SMMI", "ARE", "BE", "SAE"]]
        ctip_data = ctip_data.dropna(axis=0, how='all').dropna(axis=1, how='all').drop_duplicates()
        ctip_data_new = ctip_data.groupby("CTIP_2013 assignment").sum()
        ctip_data_removed = ctip_data_new.drop("Tract", axis=1)
        return sns.heatmap(ctip_data_removed)
     
    def frl_vs_ctip(self, frl_df_raw, block_df_dict): 
        block_database = block_df_dict.dropna(axis=0, how='all').dropna(axis=1, how='all')
        grouped_Geoid = frl_df_raw
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
    
    def gentrification_vs_demo(self, frl_df_raw, df):
        grouped_Geoid = frl_df_raw
        grouped_Geoid_filtered = grouped_Geoid[grouped_Geoid["Geoid Group"].astype('str').str.len() > 3]
        grouped_Geoid_filtered.loc[: , "New Geoid"] = grouped_Geoid_filtered.loc[: , "Geoid Group"].astype(str).str[:10].astype(int)
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
    
    def grouped_barchart(self, frl_df_raw, df):
        grouped_Geoid_gent = self.gentrification_vs_demo(frl_df_raw, df)
        gentrification_agg = grouped_Geoid_gent.groupby("Gentrification").mean()
        gentrification_agg_new = gentrification_agg.drop("4YR AVG Student Count", axis = 1)
        return [gentrification_agg_new.loc[self.gentrification_cols].plot.bar(), gentrification_agg]
        
    def stacked_barchart(self, frl_df_raw, df):
        labels = ["Student", "FRL", "AALPI", "Combo"]
#         gentrification_agg = self.grouped_barchart(frl_df_raw, df)[1]
#         print(gentrification_agg)
        grouped_Geoid_gent = self.gentrification_vs_demo(frl_df_raw, df)
        gentrification_agg = grouped_Geoid_gent.groupby("Gentrification").mean()
        plt.bar(self.gentrification_cols, gentrification_agg["4YR AVG Student Count"])
        plt.bar(self.gentrification_cols, gentrification_agg["4YR AVG FRL Count"])
        plt.bar(self.gentrification_cols, gentrification_agg["4YR AVG Eth Flag Count"])
        plt.bar(self.gentrification_cols, gentrification_agg["4YR AVG Combo Flag Count"])
        plt.legend(labels)
        
    def grouped_Geoid(self, frl_df_raw, SF_gent): 
        grouped_Geoid = self.gentrification_vs_demo(frl_df_raw, SF_gent)
        gentrification_values = []
        for i in grouped_Geoid.index: 
            val = grouped_Geoid.loc[i, "Gentrification"]
            if val == "OD":
                gentrification_values.append(1)
            elif val == "ARG":
                gentrification_values.append(2)
            elif val == "EOG":
                gentrification_values.append(3)
            elif val == "AdvG":
                gentrification_values.append(4)
            elif val == "SMMI":
                gentrification_values.append(5)
            elif val == "ARE":
                gentrification_values.append(6)
            elif val == "BE":
                gentrification_values.append(7)
            elif val == "SAE":
                gentrification_values.append(8)

        grouped_Geoid["New Gent"] = gentrification_values
        return grouped_Geoid 
    
    
    def gentrification_map(self, eligibility_classifier, final_classifier, params, grouped_Geoid):
        tiebreaker_map = final_classifier.get_tiebreaker_map(params, "geoid")
        tiebreaker_map["New Geoid"] = tiebreaker_map.index.astype(str).str[:10].astype(int)
        merged_map_tiebreaker = tiebreaker_map.merge(grouped_Geoid, left_on = "New Geoid", right_index = True)
        merged_map_tiebreaker = gpd.GeoDataFrame(merged_map_tiebreaker)
        missing_vals = tiebreaker_map[~tiebreaker_map["New Geoid"].isin(merged_map_tiebreaker["New Geoid"])]
        
        self.merged_map_tiebreaker = merged_map_tiebreaker
        self.params = params 
        self.missing_vals = missing_vals 
        return final_classifier.plot_map_new(merged_map_tiebreaker, params, missing_vals)
        
    def gentrification_map_tiebreaker(self, final_classifier):
        new_solution = final_classifier.get_solution_set(self.params)
        merged_tiebreaker_filtered = self.merged_map_tiebreaker[self.merged_map_tiebreaker.index.isin(new_solution)]
        fig, ax = plt.subplots(figsize=(15, 15))
        self.merged_map_tiebreaker.plot(ax=ax, cmap = "YlOrRd", column = "New Gent", legend = True)
        merged_tiebreaker_filtered.plot(ax=ax, alpha = 0.7, color = 'green')
        return self.missing_vals.plot(color="lightgrey", hatch = "///", label = "Missing values", ax=ax)
    
    def redlining_data(self, gent_df=SF_gent):
        redlining = gpd.read_file(REDLINING_PATH)
        points = redlining.copy()
        points.geometry = redlining['geometry'].centroid
        points.crs =redlining.crs
        new_redlining = points.join(gent_df)
        redlining_final = new_redlining[["name", "holc_id", "holc_grade", "area_description_data", "geometry", "GEOID"]]
        return redlining_final
