import sys
sys.path.append('../..')

from src.d01_data.abstract_data_api import AbstractDataApi

_block_sfha_file_path = "Census Blocks w SFHA Flag.xlsx"
_block_demographic_file_path = "Data/SF 2010 blks 022119 with field descriptions (1).xlsx"

_acs_columns = ['ACS 2013-17 est median HH income',
 'ACS 2013-17 est median family income',
 'ACS 2013-17 % aged 25+ <HS diploma',
 'ACS 2013-17 % aged 25+ HS diploma',
 'ACS 2013-17 % aged 25+ some coll no BA',
 'ACS 2013-17 % aged 25+ with Bachelors',
 'ACS 2013-17 % aged 25+ grad or prof degr',
 'ACS 2013-17 est% HH below poverty lvl',
 'ACS 2013-17 est% HH with children <17',
 'ACS 2013-17 est% aged 5-14 public sch',
 'ACS 2013-17 est% aged 5-14 pvt sch',
 'ACS 2013-17 est% aged 5-14 not in sch',
 'ACS 2013-17 est% single parent household',
 'ACS 2013-17 est% of pop foreign born',
 'ACS 2013-17 est% of foreign born not czn',
 'ACS 2013-17 est% hsng units owner occ']

class BlockDataApi(AbstractDataApi):
    """
    This class does the ETL work for the Census Block data. There are two types of Census Block data:
    - SFHA: This is block data from federal housing and household income with information flagged by
     the San Francisco Housing Authority (SFHA). Corrobarate data source.
    - Demographic: Block level demographic data.
    This data was collected in the 2010 Census and provided by SFUSD.
    """
    _cache_sfha = dict()
    _cache_demographic = dict()

    def __init__(self):
        super().__init__()
        pass
    
    def load_data(self, sfha=False):
        if sfha:
            if 'data' not in self._cache_sfha.keys():
                df_dict = self.read_data(_block_sfha_file_path)
                self._cache_sfha['fields'] = df_dict['Field Descriptions']
                self._cache_sfha['data'] = df_dict['Block data']
        else:
            if 'data' not in self._cache_demographic.keys():
                df_dict = self.read_data(_block_demographic_file_path)
                
                self._cache_demographic['fields'] = df_dict['field descriptions']
                self._cache_demographic['data'] = df_dict['block database']
                
                # Clean the Fields dataframe from NaN columns and rows:
                self._cache_demographic['fields'] = self._cache_demographic['fields'].dropna(axis=0, how='all')
                self._cache_demographic['fields'] = self._cache_demographic['fields'].dropna(axis=1, how='all')

    def get_data(self, sfha=False):
        """
        :return:
        """
        self.load_data(sfha)
        if sfha:
            df = self._cache_sfha['data'].copy()
            return df
        else:
            df = self._cache_demographic['data'].copy()
            return df
        
    def get_fields(self, sfha=False):
        """
        :return:
        """
        self.load_data(sfha)
        if sfha:
            df = self._cache_sfha['fields'].copy()
            return df
        else:
            df = self._cache_demographic['fields'].copy()
            return df
        
    def get_fields_for_columns(self, columns, sfha=False):
        if sfha:
            df = self._cache_sfha['fields'].copy()
        else:
            df = self._cache_demographic['fields'].copy()
        df.set_index('Field Name', inplace=True)
        return df.loc[columns]


if __name__ == "__main__":
    # execute only if run as a script
    block_data_api = BlockDataApi()
    df = block_data_api.get_data()
    print(df.shape)
    
    df = block_data_api.get_data(sfha=True)
    print(df.shape)
