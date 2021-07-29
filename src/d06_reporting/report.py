import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from collections import OrderedDict
from collections.abc import Iterable
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML

import sys
sys.path.append('../')

from src.d02_intermediate.classifier_data_api import ClassifierDataApi
from src.d04_modeling.knapsack_classifier import KnapsackClassifier
from src.d04_modeling.naive_classifier import NaiveClassifier
from src.d04_modeling.ctip_classifier import CtipClassifier
from src.d04_modeling.propositional_classifier import PropositionalClassifier
from src.d04_modeling.abstract_block_classifier import AbstractBlockClassifier


class Report:
    __classifier_data_api = ClassifierDataApi()
    def __init__(self, frl_key, propositional_params=None,
                 propositional_structure=None):
        self.__frl_key = frl_key
        self.__model_dict = self.initialize_models_dict(frl_key, propositional_params,
                                                        propositional_structure)
        
        
    
    def heat_map1(self, column, frl_key=None, pct_frl=True, title=""):
        if frl_key is None:
            self.__classifier_data_api.get_block_data(pct_frl=pct_frl)
        else:
            self.__classifier_data_api.get_block_data(frl_key=frl_key, pct_frl=pct_frl)
        
        self.__classifier_data_api.get_map_data()
        map_df_data = self.__classifier_data_api.get_map_df_data(cols=[column])
        map_plot = OrderedDict()
        
        display(HTML("<h3>%s</h3>" % title))
        self.__classifier_data_api.plot_map_column(map_df_data, column, cmap="YlOrRd",
                                                            save=True, legend=False, title="",
                                                            show=True)
    
    @staticmethod    
    def initialize_models_dict(frl_key, propositional_params=None, propositional_structure=None):
        if propositional_structure is None:
            propositional_structure = {'variables': [("pctAALPI", "pctFRL"), "pctBoth"],
                                    'logic': ["and", "or"]}
        if propositional_params is None:
            propositional_params = [[0.5, 0.6, x] for x in np.linspace(0, 1, num=100)]
                    
        AbstractBlockClassifier().refresh()
        model_dict = OrderedDict()
        model_dict['Naive'] = {'model': NaiveClassifier(positive_group='nFocal', rate=False, frl_key=frl_key),
                               'params': None,
                               'fname': 'naive'}
        model_dict['Naive (Prop.)'] = {'model': NaiveClassifier(positive_group='nFocal', rate=True,
                                                               frl_key=frl_key),
                                      'params': None,
                                      'fname': 'naivep'}
        model_dict['CTIP1'] = {'model': CtipClassifier(positive_group='nFocal', frl_key=frl_key),
                               'params': None,
                               'fname': 'ctip'}
        model_dict['Knapsack'] = {'model': KnapsackClassifier(positive_group='nFocal', load=True,
                                                              frl_key=frl_key, run_name=frl_key+".pkl"),
                                  'params': None,
                                  'fname': 'kc'}
        model_dict['Propositional'] = {'model': PropositionalClassifier(propositional_structure['variables'],
                                                                        propositional_structure['logic'],
                                                                        frl_key=frl_key),
                                       'params': propositional_params,
                                       'fname': 'pc'}
        print("Propositional Statement:\n%s" % model_dict['Propositional']['model'].statement)
        
        return model_dict
    
    def classifier_evalutaion_roc(self):
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
        lw = 4
        for model_name, results in results_dict.items():
            marker = None
            if len(results) < 20:
                marker = '.'
            ax.plot(results['fpr'], results['tpr'], label=model_name, linewidth=lw,
                    marker=marker, markersize=12)
        ax.set_ylabel('Proportion of focal students\n receiving priority (TPR)')
        ax.set_xlabel('Proportion of non-Focal students\n receiving priority (FPR)')
        ax.legend()
        plt.tight_layout()
        plt.savefig('outputs/roc_results_%s.png' % self.__frl_key)
        plt.show()
        
    def classifier_evalutaion_precision_recall(self):
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
        
    def classification_map(self, fpr):
        model_dict = self.__model_dict.copy()
        
        for model_name, model in model_dict.items():
            display(HTML("<h1>%s</h1>" % model_name))
            params = model['params']
            if params is None:
                 model['model'].plot_map(params=fpr, save=True, col=model['fname'])
            else:
                pass

            
        