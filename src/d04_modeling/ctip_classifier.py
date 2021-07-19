from src.d04_modeling.abstract_block_classifier import AbstractBlockClassifier
from src.d00_utils.utils import get_group_value
    
class CtipClassifier(AbstractBlockClassifier):
    
    def __init__(self, true_group='nFRL', false_group='nOther'):
        super().__init__(true_group, false_group)
        ctip13 = self.raw_data['CTIP13']
        self.solution_set = ctip13.index[ctip13 == 'CTIP1'].intersection(self.data.index)
    
    def get_roc(self):
        
        results = self.data.loc[self.solution_set].sum().to_frame().T        
        results.rename(columns={self.true_group: 'tp', self.false_group: 'fp' }, inplace=True)
        results['tpr'] = results['tp'] / self.data[self.true_group].sum()
        results['fpr'] = results['fp'] / self.data[self.false_group].sum()
        
        return results[['fpr', 'tpr']]
    
    def get_solution_set(self, fpr=None):
        return self.solution_set
        
    
    
        

        
        
        
    