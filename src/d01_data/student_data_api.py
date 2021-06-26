import pandas as pd
import sys
sys.path.append('../..')


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

        df = None
        for period in periods_list:
            df_period = self.read_data(_student_file_path.format(period=period))
            df_period.drop(columns=['Unnamed: 0'], inplace=True)
            if df is None:
                df = df_period
            else:
                df = df.append(df_period, ignore_index=True)

        return df

if __name__ == "__main__":
    # execute only if run as a script
    student_data_api = StudentDataApi()
    df = student_data_api.get_data(periods_list=["1819", "1920"])
    print(df)