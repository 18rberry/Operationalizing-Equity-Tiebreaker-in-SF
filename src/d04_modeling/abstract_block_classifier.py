import matplotlib.pyplot as plt
from collections.abc import Iterable
import numpy as np
import pandas as pd

from src.d02_intermediate.classifier_data_api import ClassifierDataApi, geoid_name, _default_frl_key
from src.d00_utils.utils import get_label, add_percent_columns, add_group_columns

_classifier_columns = ['n', 'nFRL', 'nAALPI', 'nBoth', 'nFocal']


class AbstractBlockClassifier:
    """
    Abstract class for block classification models.
    """
    map_data = None
    __classifier_data_api = ClassifierDataApi()
    
    def __init__(self, columns=None,
                 positive_group='nFocal', negative_group='nOther',
                 user=None, frl_key=_default_frl_key,
                 group_criterion=False, len_BG=8):
        """
        :param columns: columns: list of columns we want to use for the classifier
        :param positive_group: column name of the positive counts
        :param negative_group: column name of the negative counts
        :param user: not used
        :param frl_key: string that identifies which FRL data should be loaded ('tk5' or 'tk12')
        :param group_criterion: aggregate/group block data by neighborhood ('nbhd' or 'block_group')
        :param len_BG: length of block group code
        """

        self.positive_group = positive_group
        self.negative_group = negative_group
        
        # Ensure the columns list contains what we want:
        if columns is None:
            columns = [positive_group, negative_group]
        else:
            if positive_group not in columns:
                columns.append(positive_group)
            if negative_group not in columns:
                columns.append(negative_group)
        
        # Getting the raw data from classifier api:
        raw_data = self.__classifier_data_api.get_block_data(frl_key=frl_key)
        self.raw_data = raw_data
        
        # Adding the negative group column:
        raw_data[self.negative_group] = raw_data['n'] - raw_data[self.positive_group]
        
        # Extending the data with percents and block group columns:
        extended_data = add_percent_columns(raw_data)
        
        if group_criterion:
            extended_data = add_group_columns(extended_data, group_criterion, len_BG, positive_group)

        self.full_data = extended_data
        
        # Selecting columns according to the block group criterion:
        if group_criterion == "nbhd":
            data = extended_data[['n', 'Neighborhood', *columns]]
        elif group_criterion == "block_group":
            data = extended_data[['n', 'BlockGroup', *columns]]
        else:
            data = extended_data[['n', *columns]]
        
        nonan_data = data.dropna()
        self.data = nonan_data.copy()
        
        self.positive_group = positive_group
        self.negative_group = negative_group
        self.set_negative_group(positive_group, negative_group)
        
        # Initialize a prediciton and a confusion matrix dictionary (parameter tuples are keys):
        self.prediction_dict = dict()
        self.confusion_dict = dict()
        
    def set_negative_group(self, positive_group, negative_group):
        """
        Update the negative group column
        :param positive_group: column name of the positive counts
        :param negative_group: column name of the negative counts
        :return:
        """
        # can I use this to plot ROC and precision/recall curves for different definitions of focal groups?
        self.data[negative_group] = self.data['n'] - self.data[positive_group]
        
    def refresh(self):
        """
        Reset the block data
        :return:
        """
        self.__classifier_data_api.refresh()
    
    def get_solution_set(self, params):
        """
        Returns pandas.Index with blocks subset for a given parameters list.
        """
        raise NotImplementedError("get_solution_set method not implemented for abstract class")
        
    def get_roc(self, param_arr: list):
        """
        Computes the 'tpr' and 'fpr' of the classifier. Each row corresponds to a point in
        a provided parameter array.
        :param param_arr: list with parameters
        :return: pandas.DataFrame
        """
        tpr_arr = []
        fpr_arr = []
        for params in param_arr:
            fpr = self.fpr(params)
            fpr_arr.append(fpr)
            tpr = self.tpr(params)
            tpr_arr.append(tpr)

        return pd.DataFrame(data=np.array([tpr_arr, fpr_arr]).T, columns=["tpr", "fpr"])
    
    def get_precision_recall(self, param_arr: list):
        """
        Computes the 'precision' and 'recall' of the classifier. Each row corresponds to a point in
        a provided parameter array.
        :param param_arr: list with parameters
        :return: pandas.DataFrame
        """
        recall_arr = []
        precision_arr = []
        for params in param_arr:
            recall = self.recall(params)
            recall_arr.append(recall)
            precision = self.precision(params)
            precision_arr.append(precision)

        return pd.DataFrame(data=np.array([recall_arr, precision_arr]).T, columns=["recall", "precision"])
        
    def plot_roc(self, param_arr, ax=None):
        """
        Plot the ROC curve of the classifier.
        :param param_arr: list with parameters
        :param ax: matplotlib.axes._subplots.AxesSubplot
        :return:
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(25, 25))
            
        df = self.get_roc(param_arr)
        data_fpr = df["fpr"].values
        data_tpr = df["tpr"].values
        
        ax.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
        ax.plot(data_fpr, data_tpr, linestyle='solid', label="Classifier")
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title("ROC curve", fontsize=30)
        ax.legend(loc="lower right")
        plt.legend(fontsize=25)
        
        plt.show()
        
        return ax
    
    def get_heatmap(self, param_arr1: list, param_arr2=None):
        """
        Returns tuple of numpy.Array with 'fpr' and 'fnr' rates respectively, each row and column
        corresponds to a parameter in the param array.
        :param param_arr1: list with parameters
        :param param_arr2: second list with parameters (optional)
        :return: returns np.arrays with 'fpr' and 'fnr' respectivly
        """
        # In case we did not specify values for second parameter, use the same as first:
        if param_arr2 is None:
            param_arr2 = param_arr1

        heat_fpr = []
        heat_fnr = []
        # Iterate over all pairwise parameters:
        for param1 in param_arr1:
            row_fpr = []
            row_fnr = []
            for param2 in param_arr2:
                params = [param1, param2]
                row_fpr.append(self.fpr(params))
                row_fnr.append(self.fnr(params))
            heat_fpr.append(row_fpr)
            heat_fnr.append(row_fnr)

        # Build array by reversing the row order:
        heat_fpr = np.flipud(np.array(heat_fpr))
        heat_fnr = np.flipud(np.array(heat_fnr))
                
        return heat_fpr, heat_fnr
    
    def plot_heatmap(self, param_arr1, param_arr2=None, ax=None):
        """
        Returns tuple of numpy.Array with 'fpr' and 'fnr' rates respectively, each row and column
        corresponds to a parameter in the param array.
        :param param_arr1: list with parameters
        :param param_arr2: second list with parameters (optional)
        :param ax: matplotlib.axes._subplots.AxesSubplot
        :return: matplotlib.axes._subplots.AxesSubplot
        """
        
        import seaborn as sns
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(60, 30), ncols=2)
        
        if param_arr2 is None:
            param_arr2 = param_arr1
            sq = True
        else:
            sq = len(param_arr1) == len(param_arr2)
        
        heat_fpr, heat_fnr = self.get_heatmap(param_arr1, param_arr2)
        
        ax_fpr = sns.heatmap(heat_fpr,
                             vmin=0, vmax=1, cmap="YlOrRd", square=sq,
                             xticklabels=param_arr1, yticklabels=param_arr2[::-1],
                             ax=ax[0], cbar=False,
                             annot=True, annot_kws={"fontsize":20}, fmt=".2%")

        ax_fnr = sns.heatmap(heat_fnr,
                             vmin=0, vmax=1, cmap="YlOrRd", square=sq,
                             xticklabels=param_arr1, yticklabels=param_arr2[::-1],
                             ax=ax[1], cbar=False,
                             annot=True, annot_kws={"fontsize":20}, fmt=".2%")

        titles = ["False Positive Rate", "False Negative Rate"]
        for i, ax in enumerate(ax.ravel()):
            ax.set_title(titles[i], fontsize=40)
            ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=20)
            ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=20)
            
        return ax
    
    def get_tiebreaker_map(self, params, col="tiebreaker", idx_col="geoid"):

        """
        Generate SF geodataframe with tiebreaker column of focal blocks for given parameters.
        :param params: solution parameters (particular instance of the classifier)
        :param col: column name used for the tiebreaker
        :param idx_col: column to use as index to assing the tiebreaker to each block
        :return:
        """
        if self.map_data is None:
            self.map_data = self.__classifier_data_api.get_map_df_data(cols=[idx_col])
        
        map_df_data = self.map_data.copy()
        solution_set = self.get_solution_set(params)
        
        # Whether we will apply the label to a column or to the index depends on our classification (by group or by id)
        if map_df_data.index.name == idx_col:
            index = map_df_data.index.to_series()
            map_df_data[col] = index.apply(lambda x: get_label(x, solution_set, block_idx=self.data.index))
        else:
            map_df_data[col] = map_df_data[idx_col].apply(lambda x: get_label(x, solution_set))
            
        return map_df_data
    
    def plot_map(self, params, ax=None, save=False, title="", col='tiebreaker', idx_col='geoid'):
        """

        :param params: solution parameters (particular instance of the classifier)
        :param ax: matplotlib.axes._subplots.AxesSubplot
        :param save: boolean used to save the plot
        :param title: plot title
        :param col: column name used for the tiebreaker
        :param idx_col: column to use as index to assing the tiebreaker to each block (not used)
        :return: matplotlib.axes._subplots.AxesSubplot
        """
        
        map_df_data = self.get_tiebreaker_map(params, col)
        
        # if ax is None:
        #    fig, ax = plt.subplots(figsize=(4.8, 4.8))
        
        ax = self.__classifier_data_api.plot_map_column(map_df_data=map_df_data, col=col, ax=ax,
                                                        legend=False, title=title, save=save)
        return ax

    def get_confusion_matrix(self, params):
        """
        Query the confusion matrix for the given parameters
        :param params: solution parameters (particular instance of the classifier)
        :return: pandas.DataFrame
        """
        if isinstance(params, Iterable):
            params_key = tuple(params)
        else:
            params_key = params

        if params_key in self.confusion_dict.keys():
            confusion_matrix = self.confusion_dict[params_key]
        else:
            data = self.data.copy()[[self.positive_group, self.negative_group]]

            solution_set = self.get_solution_set(params)
            col = 'Real\Pred'
            data[col] = 0
            data.loc[solution_set, col] = 1

            confusion_matrix = data.groupby(col).sum()
            confusion_matrix = confusion_matrix.transpose()

            if 0 not in confusion_matrix.columns:
                confusion_matrix[0] = 0
            if 1 not in confusion_matrix.columns:
                confusion_matrix[1] = 0

            confusion_matrix = confusion_matrix[[1, 0]]
            self.confusion_dict[params_key] = confusion_matrix
            
        return confusion_matrix
    
    def fpr(self, params):
        """
        Query the FPR
        :param params: solution parameters (particular instance of the classifier)
        :return:
        """
        confusion_matrix_arr = self.get_confusion_matrix(params).values
        return confusion_matrix_arr[1,0]/(confusion_matrix_arr[1,1] + confusion_matrix_arr[1,0])
    
    def fnr(self, params):
        """
        Query the FNR
        :param params: solution parameters (particular instance of the classifier)
        :return:
        """
        confusion_matrix_arr = self.get_confusion_matrix(params).values
        return confusion_matrix_arr[0,1]/(confusion_matrix_arr[0,0] + confusion_matrix_arr[0,1])
    
    def tpr(self, params):
        """
        Query the TPR
        :param params: solution parameters (particular instance of the classifier)
        :return:
        """
        return 1 - self.fnr(params)
    
    def tnr(self, params):
        """
        Query the TNR
        :param params: solution parameters (particular instance of the classifier)
        :return:
        """
        return 1 - self.fpr(params)
    
    def recall(self, params):
        """
        Query the recall
        :param params: solution parameters (particular instance of the classifier)
        :return:
        """
        confusion_matrix_arr = self.get_confusion_matrix(params).values
        return confusion_matrix_arr[0,0]/(confusion_matrix_arr[0,0] + confusion_matrix_arr[0,1])
    
    def precision(self, params):
        """
        Query the precision
        :param params: solution parameters (particular instance of the classifier)
        :return:
        """
        confusion_matrix_arr = self.get_confusion_matrix(params).values
        return confusion_matrix_arr[0,0]/(confusion_matrix_arr[0,0] + confusion_matrix_arr[1,0])

    def intersection_tpr(self, params):

        if self.positive_group == "nBoth":
            return self.tpr(params)

        else:

            data = self.full_data.copy()[["nBoth"]]

            solution_set = self.get_solution_set(params)
            col = 'tiebreaker'
            data[col] = 0
            data.loc[solution_set, col] = 1

            grouped_data = data.groupby(col).sum()
            grouped_data_arr = grouped_data.values

            inter_tpr = grouped_data_arr[1]/(grouped_data_arr[0] + grouped_data_arr[1])

        return inter_tpr[0]