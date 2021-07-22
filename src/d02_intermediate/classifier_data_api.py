import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from time import time

from src.d01_data.block_data_api import BlockDataApi, _default_frl_key
from src.d01_data.student_data_api import StudentDataApi, _block_features, _census_block_column, \
_diversity_index_features

from src.d00_utils.utils import get_group_value

geoid_name = 'geoid'

block_data_api = BlockDataApi()

periods_list = ["1415", "1516", "1617", "1718", "1819", "1920"]
student_data_api = StudentDataApi()


class ClassifierDataApi:
    __block_data = None
    __map_data = None
    
    def __init__(self):
        pass
    
    def refresh(self):
        self.__block_data = None
    
    def get_block_data(self, frl_key=_default_frl_key):
        if self.__block_data is None:
            e = time()
            print("Loading Block FRL data...", end="")
            frl_df = self.get_frl_data(frl_key=frl_key)
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
        
        return self.__block_data.copy()
    
    def get_map_data(self):       
        if self.__map_data is None:
            geodata_path = '/share/data/school_choice/dssg/census2010/'
            file_name = 'geo_export_e77bce0b-6556-4358-b36b-36cfcf826a3c'
            data_types = ['.shp', '.dbf', '.prj', '.shx']

            sfusd_map = gpd.read_file(geodata_path + file_name + data_types[0])
            sfusd_map[geoid_name] = sfusd_map['geoid10'].astype('int64')
            sfusd_map.set_index(geoid_name, inplace=True)
            
            self.__map_data = sfusd_map.copy()
            
        return self.__map_data.copy()
    
    def get_map_df_data(self, cols):
        block_data = self.get_block_data()
        map_data = self.get_map_data()
        
        map_df_data = pd.concat([map_data.reindex(block_data.index), block_data[cols]], 
                                axis=1, ignore_index=False)
        
        return map_df_data

    @staticmethod
    def get_frl_data(frl_key=_default_frl_key):
        frl_df = block_data_api.get_data(frl=True, frl_key=frl_key).set_index('Geoid10')
        # print(frl_df)
        frl_df.index.name = geoid_name
        frl_df.columns = ['group', 'n', 'nFRL', 'nAALPI', 'nBoth']
        frl_df['nFocal'] = frl_df.apply(lambda row: row['nFRL'] + row['nAALPI'] - row['nBoth'],
                                        axis=1, raw=False)

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
        demo_df = block_data_api.get_data().set_index('Block')[['BlockGroup',
                                                                'CTIP_2013 assignment']].dropna(subset=['BlockGroup'])
        demo_df.rename(columns={'CTIP_2013 assignment': 'CTIP13'}, inplace=True)
        demo_df.index.name = geoid_name
        
        return demo_df

    @staticmethod
    def get_student_data():
        df_students = student_data_api.get_data(periods_list)
        mask = df_students[_census_block_column] == 'NaN'
        df_students.drop(df_students.index[mask], inplace=True)
        df_students[geoid_name]=df_students['census_block'].astype('int64')
        
        stud_df = df_students.groupby(geoid_name)[_diversity_index_features].agg(get_group_value)
        
        return stud_df
    
    @staticmethod
    def plot_map_column(map_df_data, col, cmap="viridis", ax=None, save=False, fig=None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(30,30))
            save = True

        map_df_data.plot(column=col, ax=ax, cmap=cmap, 
                             legend=True, legend_kwds={'orientation': "horizontal"},
                             missing_kwds={'color': 'lightgrey'})
        ax.set_title(col, fontsize=50)
        plt.tight_layout()
        plt.show()
        
        if save:
            fname = col + '.png'
            fig.savefig(fname)

        return ax
        
        
if __name__ == "__main__":
    obj = ClassifierDataApi()

    block_data = obj.get_block_data()
    map_data = obj.get_map_data()

    map_df_data = obj.get_map_df_data(['group', 'pctFRL', 'pctAALPI', 'pctBoth'])

    obj.plot_map_column(map_df_data, 'pctFRL', cmap="YlOrRd")