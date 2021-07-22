import pandas as pd
from collections.abc import Iterable

from src.d04_modeling.abstract_block_classifier import AbstractBlockClassifier, _default_frl_key


frl_columns = ["nAALPI", "nFRL", "nBoth", "nFocal", "pctAALPI", "pctFRL", "pctBoth", "pctFocal"]

######################################################################################################################

# Function to create a pre-formatted statment for logical evaluation given features, operators, and comparisors:


def get_statement(features, operators, comparisors):
    i = 0
    statement = ""
    for feature in features:
        # If there is a tuple we open a parantheses:
        if type(feature) == tuple:
            statement += " ("
            i_max = i + len(feature) - 1
            # We proceed for each entry in the tuple:
            for sub_feature in feature:
                comp = comparisors[i] + " {" + str(i) + ":.2f} "  # comparisor with the formating placeholder
                if i < i_max:
                    operator = operators[i] + " "
                else:
                    try:
                        operator = ") " + operators[i] + " "
                    except:
                        operator = ")"
                statement += sub_feature + comp + operator
                i += 1
        # No tuple means we can just do the work above:
        else:
            comp = comparisors[i] + " {" + str(i) + ":.2f} "  # comparisor with the formating placeholder
            if i < len(operators):
                operator = operators[i] + " "
            else:
                operator = ""
            statement += feature + comp + operator
            i += 1
    return statement.strip(" ")

######################################################################################################################


class PropositionalClassifier(AbstractBlockClassifier):
    def __init__(self, features, operators, comparisors=None,
                 positive_group="nFocal", negative_group="nOther",
                 user="", frl_key=_default_frl_key):
        
        columns = frl_columns
        
        self.positive_group = positive_group
        self.negative_group = negative_group
        AbstractBlockClassifier.__init__(self, columns, negative_group=self.negative_group,
                                         user=user, frl_key=frl_key)

        self.add_proposition(features, operators, comparisors)
        
    def add_proposition(self, features, operators, comparisors=None):
        '''
        No output. Adds the pre-formatted statement to class attributes. Called upon initialization.
        '''
        # Features saved as attribute:
        self.features = features
        
        # Pre-process the comparisors. If none for comparisors, assume \geq:
        if comparisors is None:
            comparisors = [" >= "]*(len(operators) + 1)
        # In case we provided a list of comparisors, ensure the spaces are there:
        else:
            for comparisor in comparisors:
                if comparisor[0] != " ":
                    comparisor = " " + comparisor
                if comparisor[-1] != " ":
                    comparisor = comparisor + " "
        self.comparisors = comparisors

        # Pre-process the operators. Make them all lower case:
        operators = [op.lower() for op in operators]
        self.operators = operators
        
        # Get the statement:
        self.statement = get_statement(features, operators, comparisors)

    def add_params(self, params):
        """
        returns string ready for evaluation and updates params attribute
        """
        if isinstance(params, Iterable):
            self.params = list(params)
        else:
            self.params = [params]
            
        return self.statement.format(*self.params)
    
    def get_solution_set(self, params):
        """
        returns pandas.Index with blocks subset for a given parameters list.
        """
        #Some pre-processing to ensure parameter key is a tuple or a float:
        if isinstance(params, Iterable):
            params_key = tuple(params)
        else:
            params_key = params
        
        #In case we have already done the predicition for these parameters, look up the dictionary:
        if params_key in self.prediction_dict.keys():
            y_hat_idx = self.prediction_dict[params_key]
        
        #Otherwise we must do the prediction from scratch:
        else:
            #Format the statement with parameters:
            s = self.add_params(params)
            #Evaluate the expression in the dataset and add it to the dictionary:
            y_hat_series = self.data.eval(s)
            #Return the index object and add it to the dictionary:
            y_hat_idx = self.data.index[y_hat_series]
            self.prediction_dict[params_key] = y_hat_idx
            
        return y_hat_idx
        

######################################################################################################################


class andClassifier(PropositionalClassifier):
    
    def __init__(self, features, comparisors=None,
                 positive_group="nFocal", negative_group="nOther",
                 user="", frl_key=_default_frl_key):
        
        operators = ["and"]*(len(features) - 1)
        PropositionalClassifier.__init__(self, features, operators, comparisors,
                                         positive_group, negative_group,
                                         user=user, frl_key=frl_key)


class orClassifier(PropositionalClassifier):
    
    def __init__(self, features, comparisors=None,
                 positive_group="nFocal", negative_group="nOther",
                 user="", frl_key=_default_frl_key):
        
        operators = ["or"]*(len(features) - 1)
        PropositionalClassifier.__init__(self, features, operators, comparisors,
                                         positive_group, negative_group,
                                         user=user, frl_key=frl_key)