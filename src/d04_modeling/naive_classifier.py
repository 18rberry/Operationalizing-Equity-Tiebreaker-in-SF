from src.d04_modeling.abstract_block_classifier import AbstractBlockClassifier, _default_frl_key


class NaiveClassifier(AbstractBlockClassifier):
    """
    Naive classifier selects focal block by ordering blocks by number or proportion of focal students in the block or
    block group.
    """

    def __init__(self, positive_group='nFRL',
                 negative_group='nOther', proportion=False,
                 frl_key=_default_frl_key,
                 group_criterion=False, len_BG=8,
                 eligibility_classifier=None, eligibility_params=[]):
        """
        :param positive_group: column name of the positive counts
        :param negative_group: column name of the negative counts
        :param proportion: boolean to trigger ordering by proportion of focal students (instead of ordering by number)
        :param frl_key: string that identifies which FRL data should be loaded ('tk5' or 'tk12')
        :param group_criterion: aggregate/group block data by neighborhood ('nbhd' or 'block_group')
        :param len_BG: length of block group code
        :param eligibility_classifier: AbstractBlockClassifier object that is used for the eligibility classification
        :param eligibility_params: parameters (if any) that must be passed to the eligibility_classifier
        """
        columns = [positive_group]
        super().__init__(columns=columns,
                         positive_group=positive_group, negative_group=negative_group,
                         frl_key=frl_key,
                         group_criterion=group_criterion, len_BG=len_BG)
        
        # Solving the naive classification problem:
        self.proportion = proportion
        self.results = self.get_results()
        
    def get_results(self):
        """
        Query results of the classifier. This method also implements the logic of the classifier
        :return: pandas.DataFrame
        """
        results = self.data.copy()
        if self.proportion:
            fun = lambda row: row[self.positive_group] / float(row['n'])
            results['proportion'] = results.apply(fun, axis=1, raw=False)
            results.sort_values(['proportion', self.positive_group], ascending=False, inplace=True)
        else:
            results.sort_values(self.positive_group, ascending=False, inplace=True)

        results['tp'] = results[self.positive_group].cumsum()
        results['fp'] = results[self.negative_group].cumsum()

        results['tpr'] = results['tp'] / self.data[self.positive_group].sum()
        results['fpr'] = results['fp'] / self.data[self.negative_group].sum()
        return results
    
    def get_roc(self, param_arr=None):
        """
        Method description is in the abstract class
        :param param_arr:
        :return:
        """
        return self.results[['fpr', 'tpr']].copy()
    
    def get_precision_recall(self, param_arr=None):
        """
        Method description is in the abstract class
        :param param_arr:
        :return:
        """
        results = self.results.copy()      
        results['recall'] = results['tp'] / self.data[self.positive_group].sum()
        results['precision'] = results['tp'] / (results['tp'] + results['fp'])

        return results

    def get_solution_set(self, fpr):
        """
        Query the solution for a particular FPR level
        :param fpr: FPR level of the solution
        :return:
        """
        mask = self.results['fpr'] <= fpr
        return self.results.index[mask]
