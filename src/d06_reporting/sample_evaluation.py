import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from collections import OrderedDict
from collections.abc import Iterable
import itertools
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
import seaborn as sns

import sys
sys.path.append('../')
# sns.set_theme(palette="pastel")

from src.d02_intermediate.classifier_data_api import ClassifierDataApi
from src.d04_modeling.knapsack_classifier import KnapsackClassifier
from src.d04_modeling.naive_classifier import NaiveClassifier
from src.d04_modeling.ctip_classifier import CtipClassifier
from src.d04_modeling.propositional_classifier import PropositionalClassifier, andClassifier, orClassifier
from src.d04_modeling.abstract_block_classifier import AbstractBlockClassifier


class SampleEvaluation:
    """
    Generate main report results
    """
    __classifier_data_api = ClassifierDataApi()

    def __init__(self, frl_key, propositional_model: PropositionalClassifier, propositional_params):
        """
        :param frl_key: string that identifies which FRL data should be loaded ('tk5' or 'tk12')
        :param propositional_params: list of parameters for the propositional classifier
        :param propositional_model: PropositionalClassifier
        """
        self.__frl_key = frl_key
        self.__model_dict = self.initialize_models_dict(frl_key, propositional_model, propositional_params)
        
    @staticmethod    
    def initialize_models_dict(frl_key, propositional_model, propositional_params):
        """
        Initialize models for report
        :param frl_key: string that identifies which FRL data should be loaded ('tk5' or 'tk12')
        :param propositional_params: list of parameters for the propositional classifier
        :param propositional_model: PropositionalClassifier
        :return:
        """
                    
        AbstractBlockClassifier().refresh()
        positive_group = propositional_model.positive_group
        model_dict = OrderedDict()
        model_dict['CTIP1'] = {'model': CtipClassifier(positive_group=positive_group, frl_key=frl_key),
                               'params': None,
                               'fname': 'ctip'}
        model_dict['Benchmark'] = {'model': NaiveClassifier(positive_group=positive_group, proportion=True,
                                                            frl_key=frl_key),
                                   'params': None,
                                   'fname': 'naivep'}
        model_dict['DSSG ET'] = {'model': propositional_model,
                                 'params': propositional_params,
                                 'fname': 'pc'}
        
        print("Propositional Statement:\n%s" % model_dict['DSSG ET']['model'].statement)
        print("Focal group: %s" % positive_group)
        return model_dict

    def heat_map1(self, column, frl_key=None, pct_frl=True, title=None, legend=False):
        """
        Generate heat map of SF for the corresponding column
        :param column: column with value used for the heat map
        :param frl_key: string that identifies which FRL data should be loaded ('tk5' or 'tk12')
        :param pct_frl: load block data with percentage columns for FRL data
        :param title: title of the plot
        :param legend: show map legend
        :return:
        """
        if frl_key is None:
            self.__classifier_data_api.get_block_data(pct_frl=pct_frl)
        else:
            self.__classifier_data_api.get_block_data(frl_key=frl_key, pct_frl=pct_frl)

        self.__classifier_data_api.get_map_data()
        map_df_data = self.__classifier_data_api.get_map_df_data(cols=[column])
        if title is not None:
            display(HTML("<h3>%s</h3>" % title))
        self.__classifier_data_api.plot_map_column(map_df_data, column, cmap="YlOrRd",
                                                   save=True, legend=legend, title="",
                                                   show=True)

    def classifier_evalutaion_roc(self, x=None):
        """
        Plot ROC curve for all the models
        :return:
        """
        model_dict = self.__model_dict.copy()

        results_dict = OrderedDict()
        for model_name, model in model_dict.items():
            params = model['params']
            if params is None:
                results_dict[model_name] = model['model'].get_roc()
            else:
                results_dict[model_name] = model['model'].get_roc(params)

        plt.rcParams['font.size'] = '10'
        fig, ax = plt.subplots(figsize=(4.8,4.8))
        lw = 2
        palette = itertools.cycle(sns.color_palette())
        for model_name, results in results_dict.items():
            markers = False
            if len(results) < 20:
                markers = True
                sns.scatterplot(ax=ax, data=results, x='fpr', y='tpr', label=model_name, color=next(palette))
            else:
                sns.lineplot(ax=ax, data=results, x='fpr', y='tpr', label=model_name, linewidth=lw,
                        markers=markers, markersize=12, color=next(palette))
        ax.set_ylabel('Proportion of focal students\n receiving priority (TPR)')
        ax.set_xlabel('Proportion of non-Focal students\n receiving priority (FPR)')
        if x is not None:
            ax.axvline(x=x, ymin=0., ymax=1., color='k', linestyle='--')
        ax.legend()
        plt.tight_layout()
        plt.savefig('outputs/roc_results_%s.png' % self.__frl_key)
        plt.show()

    def classifier_evalutaion_precision_recall(self):
        """
        Plot precision/recall curve for all the models
        :return:
        """
        model_dict = self.__model_dict.copy()

        results_dict = OrderedDict()
        for model_name, model in model_dict.items():
            params = model['params']
            if params is None:
                results_dict[model_name] = model['model'].get_precision_recall()
            else:
                results_dict[model_name] = model['model'].get_precision_recall(params)

        plt.rcParams['font.size'] = '10'
        fig, ax = plt.subplots(figsize=(4.8,4.8))
        lw = 4
        for model_name, results in results_dict.items():
            marker = None
            if len(results) < 20:
                marker = '.'
            ax.plot(results['recall'], results['precision'], label=model_name, linewidth=lw,
                    marker=marker, markersize=12)
        ax.set_xlabel('Proportion of focal students\n receiving priority (Recall)')
        ax.set_ylabel('Proportion of prioritized students\n who are focal (Precision)')
        ax.legend()
        plt.tight_layout()
        plt.savefig('outputs/precision_recall_results_%s.png' % self.__frl_key)
        plt.show()

    def classification_map(self, fpr, params):
        """
        Plot SF map with the solution/assignment for each model. Propositional classifier is not implemented.
        :param fpr: Maximum FPR of the solution
        :return:
        """
        model_dict = self.__model_dict.copy()

        for model_name, model in model_dict.items():
            display(HTML("<h1>%s</h1>" % model_name))
            use_fpr = model['params'] is None
            if use_fpr is None:
                model['model'].plot_map(params=fpr, save=True, col=model['fname'])
            else:
                model['model'].plot_map(params=params, save=True, col=model['fname'])
                
    def get_ctip1_fpr(self):
        """
        Query the false positive rate of the CTIP1 model
        :return:
        """
        model = self.__model_dict['CTIP1']['model']
        fpr = model.get_results().iloc[0]['fp'] / model.data[model.negative_group].sum()
        print("CTIP1 FPR: %.4f" % fpr)
        return fpr
    
    def get_dssg_et_params(self, fpr):
        """
        Query the parameters for the prepositional model at a given FPR.
        :param fpr:  model fpr
        :return:
        """
        model = self.__model_dict['DSSG ET']['model']
        params = self.__model_dict['DSSG ET']['params']
        roc_df = model.get_roc(params).sort_values('fpr')
        mask = roc_df['fpr'] <= fpr
        i = roc_df.index[mask][-1]
        params_fpr = params[i]
        print("Parameters DSSG ET @ %.4f:" % fpr)
        print(params_fpr)
        return params_fpr