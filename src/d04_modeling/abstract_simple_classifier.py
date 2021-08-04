from abc import ABC
from src.d04_modeling.abstract_block_classifier import AbstractBlockClassifier, _default_frl_key


class AbstractSimpleClassifier(AbstractBlockClassifier, ABC):
    def __init__(self, columns, positive_group='nFRL', negative_group='nOther', frl_key=_default_frl_key,
                 group_criterion=False, len_BG=8):
        super().__init__(columns=columns,
                         positive_group=positive_group, negative_group=negative_group,
                         frl_key=frl_key,
                         group_criterion=group_criterion, len_BG=len_BG)

    def get_results(self):
        return NotImplementedError("get_results method not implemented for abstract class")

    def get_roc(self, param_arr=None):
        """
        Method description is in the abstract class
        :param param_arr:
        :return:
        """
        results = self.get_results()
        results['tpr'] = results['tp'] / self.data[self.positive_group].sum()
        results['fpr'] = results['fp'] / self.data[self.negative_group].sum()

        return results[['fpr', 'tpr']]

    def get_precision_recall(self, param_arr: list):
        """
        Method description is in the abstract class
        :param param_arr:
        :return:
        """
        results = self.get_results()
        results['recall'] = results['tp'] / self.data[self.positive_group].sum()
        results['precision'] = results['tp'] / (results['tp'] + results['fp'])

        return results[['recall', 'precision']]