import numpy as np
import sys
sys.path.append('../..')

from src.d01_data.abstract_data_api import AbstractDataApi

_student_file_path = "Data/Cleaned/student_{period:s}.csv"

_diversity_index_col = 'Diversity Index'
_prob_col = 'prob'
_diversity_index_features = ['AALPI Score', 'Academic Score', 'Nhood SES Score', 'FRL Score']

# Henry's Index
_hoc_idx = ['HOCidx1', 'HOCidx2', 'HOCidx3']
                             
_block_features = ['freelunch_prob', 'reducedlunch_prob', 'ctip1'] + _hoc_idx + _diversity_index_features

_census_block_column = 'census_blockgroup'
_period_column = 'year'
_studentno = 'studentno'


_gamma = 2.5
def prob_focal_given_block(diversity_index):
    return np.power(diversity_index, _gamma)

def float2code(x):
    return  "%i" % x if np.isfinite(x) else "NaN"


class StudentDataApi(AbstractDataApi):
    """
    This class does the ETL work for the student data. This student data was cleaned and consolidated by the Stanford
    School choice research team and contains information from the choice process as well as student level demographic
    data.
    """

    _cache_data = dict()

    def __init__(self):
        super().__init__()
        pass
    
    def get_period_data(self, period):
        if period not in self._cache_data.keys():
            df_raw = self.read_data(_student_file_path.format(period=period))
            df_raw.rename(columns={"N'hood SES Score": 'Nhood SES Score'}, inplace=True)
            df_raw[_census_block_column] = df_raw[_census_block_column].apply(float2code)
            self._cache_data[period] = df_raw.drop(columns=['Unnamed: 0'])
        return self._cache_data[period].copy()

    def get_data(self, periods_list=None):
        """
        :param periods_list: list of periods, i.e. ["1819", "1920"]
        :return:
        """
        if periods_list is None:
            periods_list = ["1920"]

        df = None
        for period in periods_list:
            df_period = self.get_period_data(period)
            df_period[_period_column] = 2000 + int(period[-2:])
            if df is None:                
                df = df_period
            else:
                df = df.append(df_period, ignore_index=True)

        return df
    
    def get_data_by_block(self, periods_list=None):
        df = self.get_data(periods_list)
        cols = [_period_column, _census_block_column, _studentno]
        df_block = df[cols].groupby([_census_block_column, _period_column]).count()
        df_block.columns = ['count']
        indx = df_block.index
        
        cols = [_period_column, _census_block_column] + _block_features
        df_block[_block_features] = df[cols].groupby([_census_block_column, _period_column]).mean().reindex(indx).values
        
        return df_block
    
    def get_diversity_index(self, df):
        df[_diversity_index_col] = df[_diversity_index_features].mean(axis=1)
        
    def get_focal_probability(self, df):
        df[_prob_col] = df[_diversity_index_col].apply(prob_focal_given_block)


if __name__ == "__main__":
    # execute only if run as a script
    import numpy as np
    student_data_api = StudentDataApi()
    periods_list = ["1415", "1516", "1617", "1718", "1819", "1920"]
    df = student_data_api.get_data_by_block(periods_list=periods_list)
    
    block_index = df.index.get_level_values(_census_block_column).unique()

    np.random.seed(1992)
    block_ids = np.random.choice(block_index, size=5)

    print(df.loc[(blockgroup_ids, slice(None)), ['count', 'ctip1']].to_string())
    
    
