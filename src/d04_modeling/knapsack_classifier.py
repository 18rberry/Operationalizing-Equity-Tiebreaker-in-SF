import pandas as pd
from src.d04_modeling.abstract_block_classifier import AbstractBlockClassifier, _default_frl_key
from src.d04_modeling.knapsack_approx import KnapsackApprox


class KnapsackClassifier(AbstractBlockClassifier):
    
    def __init__(self, positive_group='nFRL', negative_group='nOther', load=False, frl_key=_default_frl_key,
                 run_name=None):
        columns = [positive_group]
        super().__init__(columns=columns, positive_group=positive_group,
                         negative_group=negative_group, frl_key=frl_key)
        
        # Solving the Knapsack Problem:
        data = self.data.round().astype('int64')
        self.solver = KnapsackApprox(eps=.5, data=data,
                                     value_col=positive_group,
                                     weight_col=negative_group,
                                     scale=False)
        
        if load:
            self.solver.load_value_function(fname=run_name)
        else:
            self.solver.solve()
            self.solver.save_value_function(fname=run_name)
            
    def get_results(self):
        results = self.solver.get_value_per_weight()
        
        results.reset_index(inplace=True)
        results.rename(columns={'values': 'tp', 'weights': 'fp'}, inplace=True)
        
        return results
    
    def get_roc(self, param_arr=None):
        results = self.get_results()
        results['tpr'] = results['tp'] / self.data[self.positive_group].sum()
        results['fpr'] = results['fp'] / self.data[self.negative_group].sum()
        
        return results[['fpr', 'tpr']]
    
    def get_precision_recall(self):
        results = self.get_results()      
        results['recall'] = results['tp'] / self.data[self.positive_group].sum()
        results['precision'] = results['tp'] / (results['tp'] + results['fp'])
        
        return results[['recall', 'precision']]
    
    def get_solution_set(self, fpr):
        fp = fpr * self.data[self.negative_group].sum()
        # print("False Positives Threshold: %i" % fp)
        v_opt, solution_set = self.solver.get_solution(w_max=fp)
        solution_set = pd.Index(solution_set, name=self.data.index.name)
        
        return solution_set
        
        
if __name__ == "__main__":
    model = KnapsackClassifier(positive_group='nFocal', load=True)
    results = model.get_roc()
    
    fpr = 0.1

    print(model.get_confusion_matrix(params=fpr))
    model.plot_map(params=fpr)
