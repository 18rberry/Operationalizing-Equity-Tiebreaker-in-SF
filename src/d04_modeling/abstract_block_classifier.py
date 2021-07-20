from src.d02_intermediate.classifier_data_api import ClassifierDataApi, geoid_name
from src.d00_utils.utils import get_label

classifier_data_api = ClassifierDataApi()
_classifier_columns = ['n', 'nFRL', 'nAALPI', 'nBoth', 'nFocal']

class AbstractBlockClassifier:
    map_data = None

    def __init__(self, positive_group, negative_group):
        self.positive_group = positive_group
        self.negative_group = negative_group
        raw_data = classifier_data_api.get_block_data()
        self.raw_data = raw_data
        
        data = raw_data[_classifier_columns].copy()
        data[negative_group] = data['n'] - data[positive_group]

        data.dropna(inplace=True)
        self.data = data.round().astype('int64')
        
    def get_roc(self):
        """
        returns pandas.DataFrame with 'tpr' and 'fpr' columns.
        """
        raise NotImplementedError("Method not implemented for abstract class")
        
    def get_solution_set(self, params):
        """
        returns pandas.Index with blocks subset for a given FPR.
        """
        raise NotImplementedError("Method not implemented for abstract class")
        
    def plot_map(self, params):
        if self.map_data is None:
            self.map_data = classifier_data_api.get_map_df_data(cols=['group'])
        
        map_df_data = self.map_data.copy()
        
        solution_set = self.get_solution_set(params)
        map_df_data[self.positive_group] = map_df_data.index.to_series().apply(lambda x: get_label(x, solution_set))
        
        classifier_data_api.plot_map_column(map_df_data=map_df_data, col=self.positive_group)
        
    def get_confusion_matrix(self, params):
        data = self.data.copy()
        
        solution_set = self.get_solution_set(params)
        col = self.positive_group + '_label'

        data[col] = 0
        data.loc[solution_set, col] = 1
        
        return data.groupby(col).sum()