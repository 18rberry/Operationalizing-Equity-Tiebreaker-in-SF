import pandas as pd
from src.d04_modeling.abstract_block_classifier import AbstractBlockClassifier
from src.d04_modeling.knapsack_approx import KnapsackApprox
    
class KnapsackClassifier(AbstractBlockClassifier):
    
    def __init__(self, true_group='nFRL', false_group='nOther', load=False):
        super().__init__(true_group, false_group)
        self.solver = KnapsackApprox(eps=.5, data=self.data.copy(),
                                     value_col=true_group,
                                     weight_col=false_group,
                                     scale=False)
        
        if load:
            self.solver.load_value_function()
        else:
            self.solver.solve()
            self.solver.save_value_function()
    
    def get_roc(self, fpr=None):
        
        results = self.solver.get_value_per_weight()
        
        results.reset_index(inplace=True)
        results.rename(columns={'values': 'tp', 'weights': 'fp'}, inplace=True)
        results['tpr'] = results['tp'] / self.data[self.true_group].sum()
        results['fpr'] = results['fp'] / self.data[self.false_group].sum()
        
        return results[['fpr', 'tpr']]
    
    def get_solution_set(self, fpr):
        fp = fpr * self.data[self.false_group].sum()
        # print("False Positives Threshold: %i" % fp)
        v_opt, solution_set = self.solver.get_solution(w_max=fp)
        solution_set = pd.Index(solution_set, name=self.data.index.name)
        
        return solution_set
        
        
if __name__ == "__main__":
    model = KnapsackClassifier(true_group='nFocal', load=True)
    results = model1.get_roc()
    
    fpr = 0.1

    print(model.get_confusion_matrix(fpr=fpr))
    model.plot_map(fpr=fpr)
        
    