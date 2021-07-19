from src.d04_modeling.abstract_block_classifier import AbstractBlockClassifier
from src.d04_modeling.knapsack_approx import KnapsackApprox
    
class NaiveClassifier(AbstractBlockClassifier):
    def __init__(self, true_group='nFRL', false_group='nOther', rate=False):
        self.rate = rate
        super().__init__(true_group, false_group)
        self.results = self.get_results()
        
    def get_results(self):
        results = self.data.copy()
        if self.rate:
            fun = lambda row: row[self.true_group] / float(row['n'])
            results['rate'] = results.apply(fun, axis=1, raw=False)
            results.sort_values(['rate', self.true_group], ascending=False, inplace=True)
        else:
            results.sort_values(self.true_group, ascending=False, inplace=True)

        results['tp'] = results[self.true_group].cumsum()
        results['fp'] = results[self.false_group].cumsum()

        results['tpr'] = results['tp'] / self.data[self.true_group].sum()
        results['fpr'] = results['fp'] / self.data[self.false_group].sum()
        return results
    
    def get_roc(self, fpr=None):        
        return self.results[['fpr', 'tpr']].copy()
    
    def get_solution_set(self, fpr):
        
        mask = self.results['fpr'] <= fpr
        return self.results.index[mask]
        
        

        
        
        
    