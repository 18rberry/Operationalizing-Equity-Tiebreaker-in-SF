from src.d04_modeling.abstract_block_classifier import AbstractBlockClassifier


class NaiveClassifier(AbstractBlockClassifier):
    def __init__(self, positive_group='nFRL', negative_group='nOther', rate=False, user=""):
        
        self.positive_group = positive_group
        self.negative_group = negative_group
        columns = [self.positive_group]
    
        super().__init__(columns, positive_group=self.positive_group, negative_group=self.negative_group, user=user)
        
        #Solving the naive classification problem:
        self.rate = rate
        self.results = self.get_results()
        
    def get_results(self):
        results = self.data.copy()
        if self.rate:
            fun = lambda row: row[self.positive_group] / float(row['n'])
            results['rate'] = results.apply(fun, axis=1, raw=False)
            results.sort_values(['rate', self.positive_group], ascending=False, inplace=True)
        else:
            results.sort_values(self.positive_group, ascending=False, inplace=True)

        results['tp'] = results[self.positive_group].cumsum()
        results['fp'] = results[self.negative_group].cumsum()

        results['tpr'] = results['tp'] / self.data[self.positive_group].sum()
        results['fpr'] = results['fp'] / self.data[self.negative_group].sum()
        return results
    
    def get_roc(self, param_arr=None):        
        return self.results[['fpr', 'tpr']].copy()
    
    def get_solution_set(self, fpr):
        
        mask = self.results['fpr'] <= fpr
        return self.results.index[mask]
