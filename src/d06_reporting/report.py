import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from collections import OrderedDict
from collections.abc import Iterable
import matplotlib.pyplot as plt

import sys
sys.path.append('../')

from src.d04_modeling.knapsack_classifier import KnapsackClassifier
from src.d04_modeling.naive_classifier import NaiveClassifier
from src.d04_modeling.ctip_classifier import CtipClassifier
from src.d04_modeling.propositional_classifier import PropositionalClassifier
from src.d04_modeling.abstract_block_classifier import AbstractBlockClassifier


class Report:
    
    @staticmethod
    def classifier_evalutaion_roc(frl_key, propositional_params=None, propositional_structure=None):
        if propositional_structure is None:
            propositional_structure = {'variables': [("pctAALPI", "pctFRL"), "pctBoth"],
                                    'logic': ["and", "or"]}
        if propositional_params is None:
            propositional_params = [[0.5, 0.6, x] for x in np.linspace(0, 1, num=100)]
                    
        AbstractBlockClassifier().refresh()
        model_dict = OrderedDict()
        model_dict['Naive'] = {'model': NaiveClassifier(positive_group='nFocal', rate=False, frl_key=frl_key),
                               'params': None}
        model_dict['Naive (Rate)'] = {'model': NaiveClassifier(positive_group='nFocal', rate=True,
                                                               frl_key=frl_key),
                                      'params': None}
        model_dict['CTIP1'] = {'model': CtipClassifier(positive_group='nFocal', frl_key=frl_key),
                               'params': None}
        model_dict['Knapsack'] = {'model': KnapsackClassifier(positive_group='nFocal', load=True,
                                                              frl_key=frl_key, run_name=frl_key+".pkl"),
                                  'params': None}
        model_dict['Propositional'] = {'model': PropositionalClassifier(propositional_structure['variables'],
                                                                        propositional_structure['logic'],
                                                                        frl_key=frl_key),
                                       'params': propositional_params}
        print("Propositional Statement:\n%s" % model_dict['Propositional']['model'].statement)

        results_dict = OrderedDict()
        for model_name, model in model_dict.items():
            params = model['params']
            if params is None:
                results_dict[model_name] = model['model'].get_roc()
            else:
                results_dict[model_name] = model['model'].get_roc(params)

        plt.rcParams['font.size'] = '16'
        fig, ax = plt.subplots(figsize=(10,10))
        lw = 4
        for model_name, results in results_dict.items():
            marker = None
            if len(results) < 20:
                marker = '.'
            ax.plot(results['fpr'], results['tpr'], label=model_name, linewidth=lw,
                    marker=marker, markersize=12)
        ax.set_ylabel('TPR')
        ax.set_ylabel('FPR')
        ax.legend()
        plt.tight_layout()
        plt.savefig('outputs/roc_results_%s.png' % frl_key)
        plt.show()
        