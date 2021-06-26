import pandas as pd
import numpy as np

_raw_file_path = "/share/data/school_choice/"
_out_file_path = _raw_file_path + "dssg/"

class AbstractDataApi:    
    def __init__(self):
        pass
    
    def read_dta(self, path, ratio, nrows):
        return pd.read_stata(path)
    
    def read_csv(self, path, ratio, nrows):
        if nrows is not None:
            return pd.read_csv(path, nrows=nrows, low_memory=False)
        if ratio is not None:
            return pd.read_csv(path, skiprows=lambda x: np.random.rand() > ratio, low_memory=False)
        
        df = pd.read_csv(path, low_memory=False)
        
        return df

    def read_data(self, file_name, ratio=None, nrows=None):
        file_type = file_name[-3:]
        path = _raw_file_path + file_name
            
        if file_type == "csv":
            return self.read_csv(path, ratio, nrows)
        elif file_type == "dta":
            return self.read_dta(path, ratio, nrows)
        else:
            raise Exception("Format .%s not implemented" % file_type)

    def save_data(self, df, file_name):
        df.to_csv(_out_file_path + file_name)