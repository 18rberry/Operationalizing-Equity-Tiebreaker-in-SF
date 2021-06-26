import pandas as pd

from src.d01_data.abstract_data_api import AbstractDataApi

_student_file_path = "Data/Cleaned/student_{period:s}.csv"


class StudentDataApi(AbstractDataApi):
    def __init__(self):
        super().__init__()
        pass

    def get_data(self, periods_list=None):
        """
        :param periods_list: list of periods, i.e. ["1819", "1920"]
        :return:
        """
        if periods_list is None:
            periods_list = ["1920"]

        df = pd.DataFrame()
        for period in periods_list:
            df = df.append(self.read_data(_student_file_path.format(period=period)))

        return df
