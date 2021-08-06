from src.d00_utils.utils import get_group_value
from src.d04_modeling.abstract_simple_classifier import AbstractSimpleClassifier, _default_frl_key

    
class CtipClassifier(AbstractSimpleClassifier):
    """
    CtipClassifier classifies as focal the blocks or groups of blocks with `CTIP13 == CTIP1`.
    """
    
    def __init__(self, positive_group='nFRL', negative_group='nOther', frl_key=_default_frl_key,
                 group_criterion=False, len_BG=8):
        """
        :param positive_group: column name of the positive counts
        :param negative_group: column name of the negative counts
        :param frl_key: string that identifies which FRL data should be loaded ('tk5' or 'tk12')
        :param group_criterion: aggregate/group block data by neighborhood ('nbhd' or 'block_group')
        :param len_BG: length of block group code
        """
        columns = [positive_group]
        super().__init__(columns=columns,
                         positive_group=positive_group, negative_group=negative_group,
                         frl_key=frl_key,
                         group_criterion=group_criterion, len_BG=len_BG)
        ctip13 = self.raw_data[['CTIP13', 'group']].groupby('group').agg(get_group_value)
        self.solution_set = ctip13.index[ctip13['CTIP13'] == 'CTIP1'].intersection(self.data.index)
    
    def get_results(self):
        """
        Compute the results (TP and FP) obtained by the classifier.
        :return: pandas.DataFrame
        """
        results = self.data.loc[self.solution_set].sum().to_frame().T        
        results.rename(columns={self.positive_group: 'tp', self.negative_group: 'fp'}, inplace=True)

        return results

    def get_roc(self):
        results = self.get_results()
        results['tpr'] = results['tp'] / self.data[self.positive_group].sum()
        results['fpr'] = results['fp'] / self.data[self.negative_group].sum()
        
        return results[['fpr', 'tpr']]
    
    def get_precision_recall(self):
        results = self.get_results()
        results['recall'] = results['tp'] / self.data[self.positive_group].sum()
        results['precision'] = results['tp'] / (results['tp'] + results['fp'])

        return results[['recall', 'precision']]

    def get_solution_set(self, fpr=None):
        return self.solution_set