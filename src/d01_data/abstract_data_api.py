import pandas as pd
import numpy as np

_raw_file_path = "/share/data/school_choice/"
_out_file_path = _raw_file_path + "dssg/"


class AbstractDataApi:    
    def __init__(self):
        pass

    @staticmethod
    def read_dta(path):
        return pd.read_stata(path)

    @staticmethod
    def read_csv(path):
        df = pd.read_csv(path, low_memory=False)
        
        return df

    @staticmethod
    def read_excel(path):
        df = pd.read_excel(path, sheet_name=None, engine="openpyxl")

        return df

    def read_data(self, file_name):
        file_type = file_name.split(".")[-1]
        path = _raw_file_path + file_name
            
        if file_type == "csv":
            return self.read_csv(path)
        elif file_type == "dta":
            return self.read_dta(path)
        elif file_type == "xlsx":
            return self.read_excel(path)
        else:
            raise Exception("Format .%s not implemented" % file_type)

    @staticmethod
    def save_data(df, file_name):
        df.to_csv(_out_file_path + file_name)
