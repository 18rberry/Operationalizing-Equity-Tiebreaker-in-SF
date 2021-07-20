from src.d04_modeling.abstract_block_classifier import AbstractBlockClassifier

    
class CtipClassifier(AbstractBlockClassifier):
    
    def __init__(self, positive_group='nFRL', negative_group='nOther'):
        super().__init__(positive_group, negative_group)
        ctip13 = self.raw_data['CTIP13']
        self.solution_set = ctip13.index[ctip13 == 'CTIP1'].intersection(self.data.index)
    
    def get_roc(self):
        
        results = self.data.loc[self.solution_set].sum().to_frame().T        
        results.rename(columns={self.positive_group: 'tp', self.negative_group: 'fp'}, inplace=True)
        results['tpr'] = results['tp'] / self.data[self.positive_group].sum()
        results['fpr'] = results['fp'] / self.data[self.negative_group].sum()
        
        return results[['fpr', 'tpr']]
    
    def get_solution_set(self, fpr=None):
        return self.solution_set
        
    
    
        

        
        
        
    