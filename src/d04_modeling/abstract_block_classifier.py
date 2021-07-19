from src.d02_intermediate.classifier_data_api import ClassifierDataApi, geoid_name
from src.d00_utils.utils import get_label

classifier_data_api = ClassifierDataApi()

class AbstractBlockClassifier:
    map_data = None
    def __init__(self, true_group, false_group):
        self.true_group = true_group
        self.false_group = false_group
        raw_data = classifier_data_api.get_block_data()
        self.raw_data = raw_data
        
        data = raw_data[['n', true_group, 'group']].groupby('group').sum()
        data[false_group] = data['n'] - data[true_group]
        data.dropna(inplace=True)
        self.data = data.round().astype('int64')
        
    def get_roc(self):
        """
        returns pandas.DataFrame with 'tpr' and 'fpr' columns.
        """
        raise NotImplementedError("Method not implemented for abstract class")
        
    def get_solution_set(self, fpr):
        """
        returns pandas.Index with blocks subset for a given FPR.
        """
        raise NotImplementedError("Method not implemented for abstract class")
        
    def plot_map(self, fpr):
        if self.map_data is None:
            self.map_data = classifier_data_api.get_map_df_data(cols=['group'])
        
        map_df_data = self.map_data.copy()
        
        solution_set = self.get_solution_set(fpr)
        map_df_data[self.true_group] = map_df_data['group'].apply(lambda x: get_label(x, solution_set))
        
        classifier_data_api.plot_map_column(map_df_data=map_df_data, col=self.true_group)
        
    def get_confusion_matrix(self, fpr):
        data = self.data.copy()
        
        solution_set = self.get_solution_set(fpr)
        col = self.true_group + '_label'
        data[col] = 0
        data.loc[solution_set, col] = 1
        
        return data.groupby(col).sum()