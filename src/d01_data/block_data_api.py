import numpy as np
import sys
import geopandas as gdp
sys.path.append('../..')
from src.d00_utils.utils import get_group_value
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

_hhinc_col = 'ACS 2013-17 est median HH income'
_pov_col = 'ACS 2013-17 est% HH below poverty lvl'
_bachdeg_col = 'ACS 2013-17 % aged 25+ with Bachelors'

_ses_cols = [_hhinc_col, _pov_col, _bachdeg_col]

_academic_cols = ['num of SBAC L1 scores 4-9 2015-18',
                 'num of SBAC L2 scores 4-9 2015-18',
                 'num of SBAC L3 scores 4-9 2015-18',
                 'num of SBAC L4 scores 4-9 2015-18',
                 'ttl num 4-9 test takers 2015-18']

_l1_col = 'num of SBAC L1 scores 4-9 2015-18'
_total_academic_col = 'ttl num 4-9 test takers 2015-18'

_aalpi_col = 'AALPI all TK5 stu 2017'

_tk5_stu_cols = [_aalpi_col, 'non-AALPI all TK5 stu 2017',
                 # 'DS or ML all TK5 stu 2017', 'All Others all TK5 stu 2017'
                 ]


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
        self.__cache_redline = dict()
        self.__cache_demographic = dict()
    
    def load_data(self, sfha=False, frl=False, redline=False, user=None, frl_key=_default_frl_key):
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
            
            if "data" not in self.__cache_frl[frl_key].keys():
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
        
        #Adds the redline status (HOLC grade D) to appropriate blocks:
        if redline:
            redlining_gdf_raw = gpd.read_file("/share/data/school_choice_equity/data/CASanFrancisco1937.geojson")
            redlining_gdf = redlining_gdf_raw.dissolve(by='holc_grade').to_crs(sf_gdf.crs)

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

    def get_ses_score(self):
        """
        Query SES score for each block group
        :return: pandas.DataFrame with the SES score of each 'BlockGroup' (coarser to geoid)
        """
        df = self.get_data(sfha=False).set_index('Block')

        ses_factors_max = df[_ses_cols].max()
        block_ses = df[_ses_cols + ['BlockGroup']].groupby('BlockGroup').agg(get_group_value)
        block_ses = block_ses / ses_factors_max.values[np.newaxis, :]
        block_ses.columns = ['hhinc', 'pov', 'bachdeg']

        block_ses['metric'] = 1 - block_ses['hhinc'] + block_ses['pov'] + 1 - block_ses['bachdeg']
        block_ses['score'] = block_ses['metric'] / block_ses['metric'].max()

        return block_ses

    def get_academic_score(self):
        """
        Query SES score for each block group
        :return: pandas.DataFrame with the SES score of each 'BlockGroup' (coarser to geoid)
        """
        df = self.get_data(sfha=False).set_index('Block')

        block_academics = df[_academic_cols + ['BlockGroup']].groupby('BlockGroup').sum()
        block_academics = block_academics / block_academics[_total_academic_col].values[:, np.newaxis]
        l1_score_max = block_academics[_l1_col].max()
        block_academics['score'] = block_academics[_l1_col] / l1_score_max

        return block_academics

    def get_aalpi_score(self):
        """
        Query SES score for each block group
        :return: pandas.DataFrame with the SES score of each 'BlockGroup' (coarser to geoid)
        """
        df = self.get_data(sfha=False).set_index('Block')

        block_aalpi = df[_tk5_stu_cols + ['BlockGroup']].groupby('BlockGroup').sum()

        block_aalpi['total_tk5'] = block_aalpi[_tk5_stu_cols].sum(axis=1)
        block_aalpi['aalpi_pct'] = block_aalpi[_aalpi_col] / block_aalpi['total_tk5']
        block_aalpi['score'] = block_aalpi['aalpi_pct'] / block_aalpi['aalpi_pct'].max()

        return block_aalpi
    
    def add_aa2frl(self, frl_key=_default_frl_key):
        if 'nAA' in self.__cache_frl[frl_key]['data'].columns:
            return None
        print("Adding African-American counts to FRL data...")
        block_data = self.get_data().set_index('Block')
        frl_data = self.get_data(frl=True, frl_key=frl_key)
        mask = block_data.columns.str.fullmatch(r'2017 K-5 Black')
        aa_col = block_data.columns[mask][0]
        block_data[aa_col] = block_data[aa_col].apply(lambda x: 0 if x == '--' else int(x))

        pct_aa = block_data[aa_col] / block_data[_aalpi_col]
        pct_aa.fillna(0, inplace=True)
        mask = ~pct_aa.index.duplicated()
        pct_aa = pct_aa.loc[mask]

        frl_data.set_index('Geoid10', inplace=True)
        frl_data['pctAA'] = pct_aa.reindex(frl_data.index).fillna(0).values
        frl_data['nAA'] = frl_data['pctAA'].values * frl_data['4YR AVG Eth Flag Count']
        frl_data.reset_index(inplace=True)

        self.__cache_frl[frl_key]['data'] = frl_data
        

if __name__ == "__main__":
    # execute only if run as a script
    block_data_api = BlockDataApi()
    df = block_data_api.get_data()
    print(df.shape)
    
    df = block_data_api.get_data(sfha=True)
    print(df.shape)
