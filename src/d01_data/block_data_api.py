import sys
sys.path.append('../..')

from src.d01_data.abstract_data_api import AbstractDataApi

_block_sfha_file_path = "Census Blocks w SFHA Flag.xlsx"
_block_demographic_file_path = "Data/SF 2010 blks 022119 with field descriptions (1).xlsx"


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

    def get_data(self, sfha=False):
        """
        :return:
        """
        if sfha:
            if 'data' not in self._cache_sfha.keys():
                df_dict = self.read_data(_block_sfha_file_path)
                self._cache_sfha['fields'] = df_dict['Field Descriptions']
                self._cache_sfha['data'] = df_dict['Block data']

            df = self._cache_sfha['data'].copy()
            return df
        else:
            if 'data' not in self._cache_demographic.keys():
                df_dict = self.read_data(_block_demographic_file_path)
                
                self._cache_demographic['fields'] = df_dict['field descriptions']
                self._cache_demographic['data'] = df_dict['block database']
                
                #Clean the Fields dataframe from NaN columns and rows:
                self._cache_demographic['fields'] = self._cache_demographic['fields'].dropna(axis=0, how='all')
                self._cache_demographic['fields'] = self._cache_demographic['fields'].dropna(axis=1, how='all')

            df = self._cache_demographic['data'].copy()
            return df


if __name__ == "__main__":
    # execute only if run as a script
    block_data_api = BlockDataApi()
    df = block_data_api.get_data()
    print(df.shape)
    
    df = block_data_api.get_data(sfha=True)
    print(df.shape)
