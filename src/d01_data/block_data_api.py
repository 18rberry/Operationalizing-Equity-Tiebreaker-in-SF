import sys
sys.path.append('../..')

from src.d01_data.abstract_data_api import AbstractDataApi

_block_w_sfha_file_path = "Data/Census Blocks w SFHA Flag.xlsx"
_block_demographic_file_path = "Data/SF 2010 blks 022119 with field descriptions (1).xlsx"


class BlockDataApi(AbstractDataApi):
    """
    This class does the ETL work for the Census Block data. There are two types of Census Block data:
    - with SFHA: ?
    - without SFHA: ?
    This data was collected in the 2010 Census and provided by SFUSD.
    """
    def __init__(self):
        super().__init__()
        pass

    def get_data(self, with_sfha=False):
        """
        :return:
        """
        if with_sfha:
            df = self.read_data(_block_w_sfha_file_path)
        else:
            df = self.read_data(_block_demographic_file_path)

        return df


if __name__ == "__main__":
    # execute only if run as a script
    block_data_api = BlockDataApi()
    df = block_data_api.get_data()
    print(df)
