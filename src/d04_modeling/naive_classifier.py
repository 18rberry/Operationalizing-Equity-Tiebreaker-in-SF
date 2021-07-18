from src.d04_modeling.abstract_block_classifier import AbstractBlockClassifier
from src.d04_modeling.knapsack_approx import KnapsackApprox
    
class NaiveClassifier(AbstractBlockClassifier):
    results = None
    def __init__(self, true_group='nFRL', false_group='nOther'):
        super().__init__(true_group, false_group)
    
    def get_roc(self, fpr=None):
        if self.results is None:
            results = self.data.copy().sort_values(self.true_group, ascending=False)

            results.rename(columns={self.true_group: 'tp', self.false_group: 'fp' }, inplace=True)
            results['tp'] = results['tp'].cumsum()
            results['fp'] = results['fp'].cumsum()

            results['tpr'] = results['tp'] / self.data[self.true_group].sum()
            results['fpr'] = results['fp'] / self.data[self.false_group].sum()
            self.results = results
        
        return self.results[['fpr', 'tpr']].copy()
    
    def get_solution_set(self, fpr):
        
        mask = self.results['fpr'] <= fpr
        return self.results.index[mask]
        
        

        
        
        
    