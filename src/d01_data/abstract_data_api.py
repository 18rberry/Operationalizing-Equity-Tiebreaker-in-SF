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
    
    @staticmethod
    def read_pickle(path):
        df = pd.read_pickle(path)
        
        return df

    def read_data(self, file_name, shared=True, user=""):
        if shared:
            file_type = file_name.split(".")[-1]
            path = _raw_file_path + file_name
        else:
            # In this case the file is in the dssg folder and we need to add the user name to the front:
            file_type = file_name.split(".")[-1]
            path = _out_file_path + user + "_" + file_name

        if file_type == "csv":
            return self.read_csv(path)
        elif file_type == "dta":
            return self.read_dta(path)
        elif file_type == "xlsx":
            return self.read_excel(path)
        elif file_type == "pkl":
            return self.read_pickle(path)
        else:
            raise Exception("Format .%s not implemented" % file_type)
            
    def get_data(self):
        raise NotImplementedError("Method not implemented for abstract class")

    @staticmethod
    def save_data(df, file_name):
        df.to_csv(_out_file_path + file_name)
