import pandas as pd
from src.d04_modeling.abstract_simple_classifier import AbstractSimpleClassifier, _default_frl_key
from src.d04_modeling.knapsack_approx import KnapsackApprox


class KnapsackClassifier(AbstractSimpleClassifier):
    """
    The problem of defining the focal blocks within the district can be modeled as a Knapsack problem. In this case
    the value of each item (or block) is defined by its contribution to the focal group, while the weight of each item
    (or block) is defined by its contribution to the FPR (or whatever criteria is used). The Knapsack Classifier solves
    uses the solution to this Knapsack Problem for a given FPR to find the optimal combination of focal blocks.
    """
    
    def __init__(self, positive_group='nFRL', negative_group='nOther', load=False, frl_key=_default_frl_key,
                 run_name=None):
        """
        :param positive_group: column name of the positive counts
        :param negative_group: column name of the negative counts
        :param frl_key: string that identifies which FRL data should be loaded ('tk5' or 'tk12')
        :param load: boolean to load saved Knapsack Problem solution
        :param run_name: name of saved solution
        """
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
        """
        Query results of the classifier
        :return: pandas.DataFrame
        """
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
        """
        Query the solution for a particular FPR level
        :param fpr: FPR level of the solution
        :return:
        """
        fp = fpr * self.data[self.negative_group].sum()
        # print("False Positives Threshold: %i" % fp)
        v_opt, solution_set = self.solver.get_solution(w_max=fp)
        solution_set = pd.Index(solution_set, name=self.data.index.name)
        
        return solution_set
