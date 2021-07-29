import sys
sys.path.append('../..')

from src.d01_data.abstract_data_api import AbstractDataApi
from collections import defaultdict

_block_sfha_file_path = "../school_choice_equity/data/block_sfha"
_block_demographic_file_path = "../school_choice_equity/data/block_demo"
_block_frl_file_path = {'tk5': "../school_choice_equity/data/block_frl_tk5",
                        'tk12': "../school_choice_equity/data/block_frl_tk12"}

_fields_extension = "_fields.pkl"
_data_extension = "_data.pkl"

_census_block_column = 'BlockGroup'

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

_default_frl_key = 'tk12'


class BlockDataApi(AbstractDataApi):
    """
    This class does the ETL work for the Census Block data. There are three types of Census Block data:
    - SFHA: This is block data from federal housing and household income with information flagged by
     the San Francisco Housing Authority (SFHA). Corrobarate data source.
    - Demographic: Block level demographic data. This data was collected by multiple sources.
    - FRL: Block level data on counts of focal students. Blocks with less than 5 students have been
    aggregated into block groups to avoid re-identifiability.
    """
    def __init__(self):
        super().__init__()
        self.__cache_sfha = dict()
        self.__cache_frl = defaultdict(lambda: dict())
        self.__cache_demographic = dict()
    
    def load_data(self, sfha=False, frl=False, user=None, frl_key=_default_frl_key):
        """
        Load block data
        :param sfha: boolean to load SFHA data
        :param frl: boolean to load FRL (focal student) data
        :param user: Not used anymore
        :param frl_key: string that identifies which FRL data should be loaded ('tk5' or tk12')
        :return:
        """
        if sfha:
            if 'data' not in self.__cache_sfha.keys():
                self.__cache_sfha['fields'] = self.read_data(_block_sfha_file_path + _fields_extension)
                self.__cache_sfha['data'] = self.read_data(_block_sfha_file_path + _data_extension)
        elif frl:
            if frl_key not in _block_frl_file_path.keys():
                raise Exception("Missing key for frl data")
            
            if "data" not in self.__cache_frl.keys():
                df_field = self.read_data(_block_frl_file_path[frl_key] + _fields_extension)
                
                # For grouped blocks we need to average the counts:
                df_data = self.read_data(_block_frl_file_path[frl_key] + _data_extension).copy()

                count_df = df_data.groupby(['Geoid Group']).size().to_frame(name="count").reset_index()
                extended_df = df_data.merge(count_df, on="Geoid Group")
    
                for col in list(df_data.columns)[2:]:
                    df_data[col] = extended_df[col]/extended_df["count"]
                
                self.__cache_frl[frl_key]['data'] = df_data

                # Clean the Fields dataframe from NaN columns and rows:
                df_field.dropna(axis=0, how='all', inplace=True)
                df_field.dropna(axis=1, how='all', inplace=True)
                self.__cache_frl[frl_key]['fields'] = df_field
        
        else:
            if 'data' not in self.__cache_demographic.keys():
                self.__cache_demographic['fields'] = self.read_data(_block_demographic_file_path + _fields_extension)
                self.__cache_demographic['data'] = self.read_data(_block_demographic_file_path + _data_extension)
                
                # Rename columns so that block dataframe and field dataframe match:
                self.__cache_demographic['data'] = self.__cache_demographic['data'].rename(columns={"AREA": "Area"})
                
                old = ['avg CSTELA score 2006-14', 'CSTELA test takers 2006-14']
                new = ['avg CSTELA score 2006-10', 'CSTELA test takers 2006-10']
                self.__cache_demographic['fields']['Field Name'] = \
                    self.__cache_demographic['fields']['Field Name'].replace(old, new)
                
                # Clean the Fields dataframe from NaN columns and rows:
                self.__cache_demographic['fields'] = self.__cache_demographic['fields'].dropna(axis=0, how='all')
                self.__cache_demographic['fields'] = self.__cache_demographic['fields'].dropna(axis=1, how='all')

    def get_data(self, sfha=False, frl=False, frl_key=_default_frl_key):
        """
        Query block data
        :param sfha: boolean to load SFHA data
        :param frl: boolean to load FRL (focal student) data
        :param frl_key: string that identifies which FRL data should be loaded ('tk5' or tk12')
        :return:
        """
        self.load_data(sfha=sfha, frl=frl, frl_key=frl_key)
        if sfha:
            df = self.__cache_sfha['data'].copy()
            return df
        elif frl:
            df = self.__cache_frl[frl_key]['data'].copy()
            return df
        else:
            df = self.__cache_demographic['data'].copy()
            return df
        
    def get_fields(self, sfha=False, frl=False, frl_key=_default_frl_key):
        """
        Query fields data
        :param sfha: boolean to load SFHA data
        :param frl: boolean to load FRL (focal student) data
        :param frl_key: string that identifies which FRL data should be loaded ('tk5' or tk12')
        :return:
        """
        self.load_data(sfha=sfha, frl=frl, frl_key=frl_key)
        if sfha:
            df = self.__cache_sfha['fields'].copy()
            return df
        elif frl:
            df = self.__cache_frl[frl_key]['fields'].copy()
            return df
        else:
            df = self.__cache_demographic['fields'].copy()
            return df
        
    def get_fields_for_columns(self, columns, sfha=False, frl=False, frl_key=_default_frl_key):
        """
        Query fields fpr specific columns
        :param columns:
        :param sfha: boolean to load SFHA data
        :param frl: boolean to load FRL (focal student) data
        :param frl_key: string that identifies which FRL data should be loaded ('tk5' or tk12')
        :return:
        """
        if sfha:
            df = self.__cache_sfha['fields'].copy()
        elif frl:
            df = self.__cache_frl[frl_key]['fields'].copy()
        else:
            df = self.__cache_demographic['fields'].copy()
        df.set_index('Field Name', inplace=True)
        return df.loc[columns]

    def get_classification(self, classification="first_round"):
        """
        Classify the columns of the block dataframe according to themes
        :param classification: parameter referring to classification will allow us to experiment with other
        classifications
        :return:
        """
        if not hasattr(self, "_cache_demographic"):
            self.load_data()

        if classification == "first_round":

            field_list = list(self.__cache_demographic['data'].columns)

            race_words = ["Black", "African", "Filipino", "Asian", "Hisp", "White", "Samoa", "Mixed",
                          "Pacific", "Decl", "Other", "DS", "Wht", "AALPI", "Asn", "AfAm"]
            income_words = ["income", "Lunch", "poverty"]
            housing_words = ["Hsng", "SFHA", "Hsg"]
            testing_words = ["SBAC", "CST", "test"]

            group_dict = defaultdict(list)

            for col in field_list:
                idx = field_list.index(col)
                # The first columns are identification columns:
                if idx <= field_list.index("SF Analysis Neighborhood"):
                    group_dict["ID"].append(col)
                # Then there are CTIP-related columns:
                elif idx <= field_list.index("Possible change in CTIP1 status Model 2"):
                    group_dict["CURRENT"].append(col)
                # Now let's take all the ethnicity-related columns, starting with the most
                # detailed breakdown:
                elif "F201" in col:
                    group_dict["ETHNICITY_DETAILED"].append(col)
                # The second level of detailed ethnicity (grouped for a few grades):
                elif col[6] == "-" and "stu" not in col:
                    group_dict["ETHNICITY_DETAILED_GROUP"].append(col)
                # Non-detailed ethnicity:
                elif any(word in col for word in race_words):
                    group_dict["ETHNICITY"].append(col)
                # Income-related columns:
                elif any(word in col for word in income_words):
                    group_dict["INCOME"].append(col)
                # Housing-related columns:
                elif any(word in col for word in housing_words):
                    group_dict["HOUSING"].append(col)
                # Testing-related columns:
                elif any(word in col for word in testing_words):
                    group_dict["TEST"].append(col)
                # All others are general demographic info:
                else:
                    group_dict["DEMOGRAPHIC"].append(col)
            
            return group_dict

        else:
            print("This classification has not been defined")
            return None


if __name__ == "__main__":
    # execute only if run as a script
    block_data_api = BlockDataApi()
    df = block_data_api.get_data()
    print(df.shape)
    
    df = block_data_api.get_data(sfha=True)
    print(df.shape)
