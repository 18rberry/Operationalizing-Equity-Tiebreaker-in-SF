import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from time import time

from src.d01_data.block_data_api import BlockDataApi, _default_frl_key
from src.d01_data.student_data_api import StudentDataApi, _block_features, _census_block_column, \
_diversity_index_features

from src.d00_utils.utils import get_group_value, add_percent_columns

geoid_name = 'geoid'

block_data_api = BlockDataApi()

periods_list = ["1415", "1516", "1617", "1718", "1819", "1920"]
student_data_api = StudentDataApi()

#Those are the demographic columns we want:
block_columns = ['BlockGroup','CTIP_2013 assignment','SF Analysis Neighborhood','SFHA_ex_Sr']
block_columns_rename = {'CTIP_2013 assignment': 'CTIP13',
                        'SF Analysis Neighborhood':'Neighborhood',
                        'SFHA_ex_Sr':'Housing'}

frl_column_dict = {'Geoid Group': 'group', '4YR AVG Student Count': 'n', '4YR AVG FRL Count': 'nFRL',
       '4YR AVG Eth Flag Count': 'nAALPI', '4YR AVG Combo Flag Count': 'nBoth'}

class ClassifierDataApi:
    __block_data = None
    __redline_data = None
    __map_data = None
    
    def __init__(self):
        pass
    
    def refresh(self):
        """
        Reset the block data
        :return:
        """
        self.__block_data = None
    
    def get_block_data(self, redline=True, frl_key=_default_frl_key, pct_frl=False):
        """
        Query block data from all three sources.
        :param frl_key: string that identifies which FRL data should be loaded ('tk5' or tk12')
        :param pct_frl: boolean to add the percent values of the frl variables
        :return:
        """
        if self.__block_data is None:
            e = time()
            print("Loading Block FRL data...", end="")
            frl_df = self.get_frl_data(frl_key=frl_key)
            if pct_frl:
                frl_df = add_percent_columns(frl_df)
            print("%.4f" % (time()-e))
            e = time()
            print("Loading Block Demographic data...", end="")
            demo_df = self.get_demo_data()
            print("%.4f" % (time()-e))
            e = time()
            print("Loading Student Demographic data...", end="")
            stud_df = self.get_student_data()
            print("%.4f" % (time()-e))

            df = pd.concat([demo_df,
                            stud_df.reindex(demo_df.index),
                            frl_df.reindex(demo_df.index)],
                           axis=1,
                           ignore_index=False)
            
            self.__block_data = df
            
            #Add the redline status:
            if redline:
                block_gdf = self.get_map_df_data(cols="BlockGroup")
                self.__block_data["Redline"] = self.get_redline_status(block_gdf)
        
        return self.__block_data.copy()
    
    def get_map_data(self):
        """
        Query map data used to build the geographic maps
        """
        if self.__map_data is None:
            geodata_path = '/share/data/school_choice/dssg/census2010/'
            file_name = 'geo_export_e77bce0b-6556-4358-b36b-36cfcf826a3c'
            data_types = ['.shp', '.dbf', '.prj', '.shx']

            sfusd_map = gpd.read_file(geodata_path + file_name + data_types[0])
            sfusd_map[geoid_name] = sfusd_map['geoid10'].astype('int64')
            sfusd_map.set_index(geoid_name, inplace=True)
            
            self.__map_data = sfusd_map
            
        return self.__map_data
    
    def get_redline_map_data(self):
        """
        Query HOLC grades map data used in the redline criterion
        """
        if self.__redline_data is None:
            geodata_path = '/share/data/school_choice_equity/data/'
            file_name = 'CASanFrancisco1937'
            data_type = '.geojson'

            redline_map = gpd.read_file(geodata_path + file_name + data_type)
            self.__redline_data = redline_map
            
        return self.__redline_data
    
    def get_map_df_data(self, cols):
        """
        Append block data to the map data geopandas.DataFrame
        :param cols: Columns from block data that should be appended to the map data
        :return:
        """
        block_data = self.get_block_data()
        map_data = self.get_map_data()
        
        if cols == [geoid_name]:
            map_df_data = map_data.reindex(block_data.index)

        else:
            map_df_data = pd.concat([map_data.reindex(block_data.index), block_data[cols]],
                                     axis=1, ignore_index=False)
        
        return map_df_data

    @staticmethod
    def get_frl_data(frl_key=_default_frl_key):
        """
        Query FRL data
        :param frl_key: string that identifies which FRL data should be loaded ('tk5' or tk12')
        :return:
        """
        block_data_api.load_data(frl=True, frl_key=frl_key)
        block_data_api.add_aa2frl(frl_key=frl_key)
        frl_df = block_data_api.get_data(frl=True, frl_key=frl_key).set_index('Geoid10')
        # print(frl_df)
        frl_df.index.name = geoid_name
        frl_df.rename(columns=frl_column_dict, inplace=True)
        frl_df['nFocal'] = frl_df.apply(lambda row: row['nFRL'] + row['nAALPI'] - row['nBoth'],
                                        axis=1, raw=False)
        frl_df['nAAFRL'] = frl_df.apply(lambda row: row['nBoth'] * row['pctAA'], axis=1)

        # we want to find the blocks that share a group index
        mask = frl_df['group'] < 1000
        last_group_index = frl_df.loc[mask, 'group'].max()
        # then we generate a new set of group indexes for the standalone blocks that is more coherent 
        # with the indexes of the grouped blocks
        num_of_new_indexes = np.sum(~mask)
        new_group_index = np.arange(num_of_new_indexes) + 1 + last_group_index

        frl_df.at[~mask, 'group'] = new_group_index
        frl_df['group'] = frl_df['group'].astype('int64')
        
        return frl_df

    @staticmethod
    def get_demo_data():
        """
        Query demographic data
        :return:
        """
        
        #Collect the meaningful demographic columns:
        demo_df = block_data_api.get_data().set_index('Block')[block_columns].dropna(subset=['BlockGroup'])
        #Clean the SFHA column:
        demo_df = demo_df.replace({'SFHA_ex_Sr': {'yes': True, 'no': False}})
        
        #Rename the columns for easier acces:
        demo_df.rename(columns=block_columns_rename, inplace=True)
        
        #Set index as geoid
        demo_df.index.name = geoid_name
        
        return demo_df

    @staticmethod
    def get_student_data():
        """
        Query student data
        :return:
        """
        df_students = student_data_api.get_data(periods_list)
        mask = df_students[_census_block_column] == 'NaN'
        df_students.drop(df_students.index[mask], inplace=True)
        df_students[geoid_name]=df_students['census_block'].astype('int64')
        
        stud_df = df_students.groupby(geoid_name)[_diversity_index_features].agg(get_group_value)
        
        return stud_df
    
    def get_redline_status(self, map_data):
        """
        Appends to the block dataframe the redline status i.e. whether the block was in a grade D HOLC area
        :param block_df: block GeoDataFrame indexed by geoids with geometries of each block
        :return: pandas (Boolean) Series of whether block intersects redline geometries
        """
        
        if self.__redline_data is None:
            redline_df = self.get_redline_map_data()
            
        redlining_by_grade = self.__redline_data.dissolve(by='holc_grade').to_crs(map_data.crs)
        
        redline_series = map_data.buffer(0).intersects(redlining_by_grade['geometry']['D'].buffer(0), align=False)
        
        return redline_series
    
    @staticmethod
    def plot_map_column(map_df_data, col, missing_vals=False, cmap="viridis", ax=None, save=False,
                        fig=None, title=None, legend=True, show=True):
        """
        Plot map data with color set to the columns `col`
        :param map_df_data: geopandas.DataFrame of SFUSD
        :param col: column of `map_df_data` with the value of interest
        :param cmap: color map for the plot
        :param ax: (optional) axis values. Must also provide `fig` value
        :param save: boolean to save the plot
        :param fig: (optional) figure values. Must also provide `ax` value
        :param title: title of the figure
        :param legend: boolean to show legend of the plot
        :param show: boolean to show the figure
        :return:
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(10,10))
            save = True
        
        #Missing values workaround for the gentrification plot:
        if missing_vals:
            map_df_data.plot(column=col, ax=ax, cmap=cmap, 
                             legend=legend, legend_kwds={'orientation': "horizontal"})
            missing_vals.plot(color="lightgrey", hatch = "///", label = "Missing values", ax=ax)
            
        else:
            map_df_data.plot(column=col, ax=ax, cmap=cmap, 
                             legend=legend, legend_kwds={'orientation': "horizontal"},
                             missing_kwds={'color': 'lightgrey'})
        
        if title is None:
            ax.set_title(col, fontsize=12)
        else:
            ax.set_title(title, fontsize=12)
        if show:
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        if save:
            fname = 'outputs/' + col + '.png'
            fig.savefig(fname)

        return ax

    @staticmethod
    def plot_map_column_new(map_df_data, col, cmap="YlOrRd", ax=None, save=False, fig=None,
                            title=None, legend=True, show=True):
        if ax is None:
            fig, ax = plt.subplots(figsize=(4.8,4.8))
            save = True
        map_df_data.plot(column=col, ax=ax, cmap=cmap, marker = 'o', color = 'black',
                         legend=legend, legend_kwds={'orientation': "horizontal"},
                         missing_kwds={'color': 'lightgrey'})
        if title is None:
            ax.set_title(col, fontsize=12)
        else:
            ax.set_title(title, fontsize=12)
        if show:
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        if save:
            fname = 'outputs/' + col + '.png'
            fig.savefig(fname)

        return ax
        
        
if __name__ == "__main__":
    obj = ClassifierDataApi()

    block_data = obj.get_block_data()
    map_data = obj.get_map_data()

    map_df_data = obj.get_map_df_data(['group', 'pctFRL', 'pctAALPI', 'pctBoth'])

    obj.plot_map_column(map_df_data, 'pctFRL', cmap="YlOrRd")