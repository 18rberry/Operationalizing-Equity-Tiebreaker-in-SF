import sys
sys.path.append('../../')

import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def get_statement(features, operators, comparisors):
    i = 0
    statement = ""
    
    for feature in features:
        #If there is a tuple we open a parantheses:
        if type(feature) == tuple:
            
            statement += " ("
            i_max = i + len(feature) - 1
            
            #We proceed for each entry in the tuple:
            for sub_feature in feature:
                comp = comparisors[i] + " {" + str(i) + ":.2f} " #comparisor with the formating placeholder
                if i < i_max:
                    operator = operators[i] + " "
                else:
                    try:
                        operator = ") " + operators[i] + " "
                    except:
                        operator = ")"
                
                statement += sub_feature + comp + operator
                i += 1
        
        #No tuple means we can just do the work above:
        else:
            comp = comparisors[i] + " {" + str(i) + ":.2f} " #comparisor with the formating placeholder
            if i < len(operators):
                operator = operators[i] + " "
            else:
                operator = ""
            
            statement += feature + comp + operator
            i += 1

    return statement.strip(" ")

def load_map(idx='geoid10'):
    
    #JUAN: Where is the code that leaves the water grey?
    geodata_path = '/share/data/school_choice/dssg/census2010/'
    file_name = 'geo_export_e77bce0b-6556-4358-b36b-36cfcf826a3c'
    data_types = ['.shp', '.dbf', '.prj', '.shx']
    sfusd = gpd.read_file(geodata_path + file_name + data_types[0])

    mask = sfusd['intptlon10'] < '-122.8'
    mask &= sfusd['awater10'] == 0.0
    # get rid of water

    sfusd = sfusd.set_index(idx)

    return sfusd

####################################################################################################

class Propositional_Classifier:
    
    def __init__(self, features, operators, comparisors=None):
        
        self.features = features
        
        #Pre-process the comparisors. If none for comparisors, assume \geq:
        if comparisors is None:
            comparisors = [" >= "]*(len(operators) + 1)
        self.comparisors = comparisors
        
        #Pre-process the operators. Make them all lower case:
        operators = [op.lower() for op in operators]
        self.operators = operators
        
        #Get the statement:
        self.statement = get_statement(features, operators, comparisors)
        
        #Initialize a prediciton and a confusion dictionary (parameter tupels are keys):
        self.prediction_dict = dict()
        self.confusion_dict = dict()
    
    def add_params(self, params):
        self.params = params
        return self.statement.format(*params)
    
    def predict(self, data, params):        
        
        s = self.add_params(params)
        
        #Initialize prediciton array and evaluations:
        y_hat_list = []
        fp, tp, fn, tn = 0, 0, 0, 0
        
        #We need to replace the columns with % first since the absolute columns have a substring of those:
        first_columns = [c for c in data.columns if "(%)" in c]
        other_columns = list(set(data.columns) - set(first_columns))
        
        #RMK: Could improve this runtime. Maybe define the eval as a function that takes columns of df rather than 
        #       using iterrows? Confusion matrix is created here to avoid doing iteration twice. Fine for a single
        #       prediction but tedious for heatmap/ROC computation.
        for block_row in data.iterrows():
            
            block = block_row[1]
            
            instructions1 = block[first_columns].to_dict()
            instructions2 = block[other_columns].to_dict()
            
            focal_students = block["AALPI"] + block["FRL"] - block["BOTH"]
            
            s_eval = s
            for key in instructions1.keys():
                s_eval = s_eval.replace(key, str(instructions1[key]))
            for key in instructions2.keys():
                s_eval = s_eval.replace(key, str(instructions2[key]))
            
            y_hat = eval(s_eval)
            
            #Using the prediction, update confusion matrix:
            if y_hat == False:
                fn += focal_students
                tn += block["COUNT"] - focal_students
            else:
                fp += block["COUNT"] - focal_students
                tp += focal_students
            
            y_hat_list.append(y_hat)
        
        prediction = np.array(y_hat_list)
        confusion_matrix = np.array([[tp, fn], [fp, tn]])
        
        self.prediction_dict[tuple(params)] = prediction
        self.confusion_dict[tuple(params)] = confusion_matrix
        
        return prediction
    
    def get_sol_df(self, data, y_hat):
        
        sol_df = data[["Geoid10"]].copy()
        sol_df["tiebreaker"] = y_hat
        
        self.sol = sol_df
        
        return sol_df
    
    def get_rates(self, params, which=None):
        
        rates = dict()
        
        try:
            cm = self.confusion_dict[tuple(params)]
        
        except:
            print("Predict result first")
            return (None, None)
        
        rates["fpr"] = cm[1,0]/(cm[1,0] + cm[1,1])
        rates["fnr"] = cm[0,1]/(cm[0,1] + cm[0,0])
        rates["tpr"] = cm[0,0]/(cm[0,1] + cm[0,0])
        rates["tnr"] = cm[1,1]/(cm[1,0] + cm[1,1])
        
        if which is None:
            return rates
        
        else:
            res = []
            for rate in which:
                res.append(rates[rate])
        
            return tuple(res)
    
    def get_multiple_rates2D(self, data, which, params_range=(0,1), n_params=10):
        
        val=np.linspace(params_range[0], params_range[1], num=n_params)
        
        data_dict = dict()
        for rate in which:
            data_dict[rate] = []
        
        for p1 in val:
            for p2 in val:
                params = [p1, p2]
                
                if tuple(params) not in self.confusion_dict:
                    y_hat = self.predict(data, params)
                
                rates_tuple = self.get_rates(params, which)
                
                for rate, rate_matrix in zip(which, rates_tuple):
                    data_dict[rate].append(rate_matrix)
        
        for rate in which:
            data_dict[rate] = np.reshape(np.array(data_dict[rate]), (len(val), len(val)))
        
        res = []
        for rate in which:
            res.append(data_dict[rate])
        
        return tuple(res)
    
    def get_multiple_rates1D(self, data, which, params_range=(0,1), n_params=10):
        
        val=np.linspace(params_range[0], params_range[1], num=n_params)
        
        data_dict = dict()
        for rate in which:
            data_dict[rate] = []
        
        for p1 in val:
            
            params = [p1]
            
            if tuple(params) not in self.confusion_dict:
                y_hat = self.predict(data, params)
                
            rates_tuple = self.get_rates(params, which)
                
            for rate, rate_matrix in zip(which, rates_tuple):
                data_dict[rate].append(rate_matrix)
        
        for rate in which:
            data_dict[rate] = np.array(data_dict[rate])
        
        res = []
        for rate in which:
            res.append(data_dict[rate])
        
        return tuple(res)
    
    def display_heat(self, data, params_range=(0,1), n_params=10, ax=None):
        
        if len(self.features) > 2:
            print("Heatmap only defined for two-dimensional parameter spaces")
            return None
        
        data_fpr, data_fnr = self.get_multiple_rates2D(data, ["fpr", "fnr"], params_range, n_params)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(60,30), ncols=2)
        
        val = np.linspace(params_range[0], params_range[1], num=n_params)
        
        ax_fpr = sns.heatmap(data_fpr,
                             vmin=params_range[0], vmax=params_range[1], cmap="YlOrRd", square=True,
                             xticklabels=val, yticklabels=val,
                             ax=ax[0], cbar=False, annot=True)
        ax_fnr = sns.heatmap(data_fnr,
                             vmin=params_range[0], vmax=params_range[1], cmap="YlOrRd", square=True,
                             xticklabels=val, yticklabels=val,
                             ax=ax[1], cbar=False, annot=True)
        
        titles = ["False Positive Rate", "False Negative Rate"]
        for i, axis in enumerate(ax.ravel()):
            axis.set_title(titles[i], fontsize=30)
        
        return ax
    
    def display_map(self, data, y_hat, idx='geoid10', ax=None):
        sfusd = load_map()
        
        sol_df = self.get_sol_df(data, y_hat)
        sfusd_sol = sfusd.merge(sol_df, left_on='geoid10', right_on='Geoid10')
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(30,30))
            
        cmap = mpl.colors.ListedColormap(["yellow", "red"])
        
        ax = sfusd_sol.plot(column="tiebreaker", ax=ax, cmap=cmap,
                            legend=False,
                            missing_kwds={'color': 'lightgrey'})
        
        ax.set_title(self.add_params(self.params), fontsize=50)
        
        plt.show()
        
        return ax
        
    def display_ROCcurve(self, data, params_range=(0,1), n_params=10, ax=None):
        
        if len(self.features) > 1:
            print("ROC curve only defined for one-dimensional parameter spaces")
            return None
        
        data_fpr, data_tpr = self.get_multiple_rates1D(data, ["fpr", "tpr"], params_range, n_params)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(25,20))
        
        ax.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
        ax.plot(data_fpr, data_tpr, marker='.', label="Classifier")
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(self.statement, fontsize=30)
        ax.legend()
        
        plt.show()
        
        return ax
    
    def display_gentrification(self):
        #ADDING IT FROM RIYA'S NOTEBOOK!
        pass
    
class AND_Classifier(Propositional_Classifier):
    
    def __init__(self, features, comparisors=None):
        operators = ["and"]*(len(features) - 1)
        Propositional_Classifier.__init__(self, features, operators, comparisors)

class OR_Classifier(Propositional_Classifier):
    
    def __init__(self, features, comparisors=None):
        operators = ["or"]*(len(features) - 1)
        Propositional_Classifier.__init__(self, features, operators, comparisors)