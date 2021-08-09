import pandas as pd
import numpy as np
from IPython.core.display import display, HTML
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from src.d00_utils.utils import get_group_value

sns.set_theme(style="ticks", palette="pastel")
pd.set_option('display.max_rows', None)
figsize = (6.8,5.2)

def default_is_focal(row):
    return (row['FRL'] + row['AALPI']) > 1


def q1(x):
    return np.quantile(x, .25)


def q2(x):
    return np.quantile(x, .5)


def q3(x):
    return np.quantile(x, .75)


def topkrank(x, k):
    return np.nanmean(x <= k)


def countnan(x):
    return np.isnan(x).sum()


class SimulationEvaluation:
    """
    Methods to evaluate simulation results for multiple methods given a policy.
    """
    __diversity_category_col = 'Diversity_Category3'  # diversity category column
    __program_cutoff = 'program_cutoff'  # program cutoff column (in points)
    __cutoff_tiebreaker = 'cutoff_tiebreaker'  # the cutoff tiebreaker of the program to which the student was assigned
    __focal_block = 'focal_block'  # if students gets equity tiebreaker
    __tiebreaker_status = 'status'  # if student counts as TP, TN, FP or FN
    __focal_label = 'focal'
    __equity_tiebreaker_list = None
    __num_iterations = None
    __filename_template = None
    __rank_results_df = None
    __summary_dict = None
    __tiebreaker_status_ordering = ['FN', 'TN', 'TP', 'FP1', 'FP0']

    def __init__(self, is_focal=None,
                 student_data_file="/share/data/school_choice_equity/simulator_data/student/drop_optout_{}.csv",
                 period="1819",
                 assignments_dir="/share/data/school_choice_equity/local_runs/Assignments/"):
        """

        :param is_focal: function that labels each student as focal given the student features
        :param student_data_file: path from which to import the augmented student data used for the simulations
        :param period: student data period
        :param assignments_dir: path from which to import the assigment results
        """
        if is_focal is None:
            is_focal = default_is_focal
        student_df = pd.read_csv(student_data_file.format(period)).set_index('studentno')
        # mask = student_df['grade'] == 'KG'
        # student_df = student_df.loc[mask]
        student_df['AA'] = (student_df['resolved_ethnicity'] == 'Black or African American').astype('int64')
        student_df[self.__focal_label] = student_df.apply(lambda row: is_focal(row), axis=1).astype('int64')
        self.__student_df = student_df
        self.__assignment_dir = assignments_dir

    def set_simulation_config(self, equity_tiebreaker_list, num_iterations, policy="Medium1", guard_rails=0):
        """
        Set the simulation configuration we want to evaluate.
        :param equity_tiebreaker_list: list of the methods we want to evaluate
        :param num_iterations: number of iterations from simulation
        :param policy: policy name ('Medium1' or 'Con1')
        :param guard_rails: policy includes guardrails (0: yes and -1: no)
        :return:
        """
        self.__equity_tiebreaker_list = equity_tiebreaker_list
        self.__num_iterations = num_iterations
        self.__filename_template = self.get_filename_template(policy, guard_rails)

    def get_rank_df(self):
        """
        Get the rank results dataframe
        :return:
        """
        return self.__rank_results_df.copy()

    def get_summary_df(self):
        """
        Get the summary dataframe
        :return:
        """
        return self.__summary_dict.copy()

    @staticmethod
    def get_specific_program_cutoff(x, diversity_category: int):
        """
        Query the program cutoff for student of a particular diversity category
        :param x: values of program_cutoff column
        :param diversity_category: student diversity category
        :return:
        """
        if isinstance(x, float):
            return x
        x_list = x[1:-1].split()
        return float(x_list[diversity_category])

    def check_tiebreaker(self, row):
        """
        Decode which tiebreaker a student has by inspecting the priority score
        :param row: row from student assignment data
        :return:
        """
        if self.__diversity_category_col in row.index:
            diversity_category = row[self.__diversity_category_col]
            cut_off = self.get_specific_program_cutoff(row[self.__program_cutoff], diversity_category)
        else:
            cut_off = row[self.__program_cutoff]

        if cut_off > 4:
            return "sibiling"
        elif cut_off > 3:
            return "equity+zone"
        elif cut_off > 2:
            return "equity"
        elif cut_off > 1:
            return "zone"
        elif cut_off > 0:
            return "lottery"
        else:
            return "none"

    def augment_assigment(self, assignment_df: pd.DataFrame, equity_tiebreaker):
        """
        Add additional columns to the assigment data.
        :param assignment_df: assignment data
        :param equity_tiebreaker: method used as equity tiebreaker
        :return:
        """
        if equity_tiebreaker == 'none':
            self.__student_df[equity_tiebreaker] = 0.
        elif equity_tiebreaker == 'test':
            self.__student_df[equity_tiebreaker] = self.__student_df['ctip1']
        assignment_df[self.__cutoff_tiebreaker] = assignment_df.apply(lambda row: self.check_tiebreaker(row), axis=1,
                                                                      raw=False)
        assignment_df[self.__focal_block] = self.__student_df[equity_tiebreaker].reindex(assignment_df.index)
        assignment_df[self.__focal_label] = self.__student_df[self.__focal_label].reindex(assignment_df.index)
        assignment_df['AALPI'] = self.__student_df['AALPI'].reindex(assignment_df.index)
        assignment_df['FRL'] = self.__student_df['FRL'].reindex(assignment_df.index)
        self.get_student_tiebreaker_status(assignment_df)

    def get_student_tiebreaker_status(self, df: pd.DataFrame):
        """
        Add column to assigment data to determine if a student is a True Positive (TP), True Negative (TN),
        False Positive (FP) or False Negative (FN).
        :param df: assigment data
        :return:
        """
        mask_focal = df[self.__focal_label] == 1
        mask_focal_block = df[self.__focal_block] == 1
        mask_extended_focal = df['AALPI'] | df['FRL']
        df[self.__tiebreaker_status] = "TN"
        df.at[mask_focal & mask_focal_block, self.__tiebreaker_status] = "TP"
        df.at[(~mask_focal & mask_focal_block) & mask_extended_focal, self.__tiebreaker_status] = "FP1"
        df.at[(~mask_focal & mask_focal_block) & ~mask_extended_focal, self.__tiebreaker_status] = "FP0"
        df.at[mask_focal & ~mask_focal_block, self.__tiebreaker_status] = "FN"

    @staticmethod
    def get_filename_template(policy, guard_rails):
        """
        Query the filename template for a particular policy and guard_rails setting
        :param policy: policy form simulation ('Medium1' or 'Con1')
        :param guard_rails: guard rails option (0 or -1)
        :return:
        """
        if policy == "Con1":
            if guard_rails == 0:
                return "Assignment_{}_CTIP1_round_merged123_policyCon1GuardRails0-RealPref_tiesSTB_prefExtend0_iteration{}.csv"
            else:
                return "Assignment_{}_CTIP1_round_merged123_policyCon1-RealPref_tiesSTB_prefExtend0_iteration{}.csv"
        elif policy == "Medium1":
            if guard_rails == 0:
                return "Assignment_{}_CTIP1_round_merged123_policyMedium1GuardRails0-RealPref_tiesSTB_prefExtend0_iteration{}.csv"
            else:
                return "Assignment_{}_CTIP1_round_merged123_policyMedium1-RealPref_tiesSTB_prefExtend0_iteration{}.csv"
        raise Exception("Configuration: (policy: %s, guard_rails: %s) is not available" % (policy, guard_rails))

    def get_summary_iteration(self, assignment_df, equity_tiebreaker, iteration):
        """
        Generate summary statistics of assigment data for a particular iteration
        :param assignment_df: assigment data
        :param equity_tiebreaker: method used as equity tiebreaker
        :param iteration: iteration
        :return:
        """
        assignment_df['iteration'] = iteration
        self.augment_assigment(assignment_df, equity_tiebreaker)
        evaluation_columns = [self.__diversity_category_col, 'rank', 'designation', 'In-Zone Rank',
                              self.__cutoff_tiebreaker, self.__tiebreaker_status, self.__focal_label, 'iteration']

        group_columns = ['iteration', self.__diversity_category_col, self.__focal_label]

        rank_funs = ['count', 'mean', 'min', q1, q2, q3, 'max']
        return assignment_df[evaluation_columns].groupby(group_columns).agg({'rank': rank_funs})

    def query_summary_df(self):
        """
        Compute summary statistics of assigment data
        :return:
        """
        summary_dict = dict()
        for equity_tiebreaker in self.__equity_tiebreaker_list:
            summary_df = []
            for iteration in range(self.__num_iterations):
                filename = self.__filename_template.format(equity_tiebreaker, iteration)
                assignment_df = pd.read_csv(self.__assignment_dir+filename).set_index('studentno')
                assignment_df['iteration'] = iteration
                summary_df += [self.get_summary_iteration(assignment_df, equity_tiebreaker, iteration)]

            # group_columns = [self.__diversity_category_col, self.__focal_label, 'tiebreaker']
            summary_df = pd.concat(summary_df, axis=0).groupby([self.__diversity_category_col, self.__focal_label]).mean()

            summary_dict[equity_tiebreaker] = summary_df

            self.__summary_dict = summary_dict

    def display_summary_df(self):
        """
        Display summary statistics
        :return:
        """
        display(HTML("<h3>Results grouped by focal</h3>" ))
        for equity_tiebreaker in self.__equity_tiebreaker_list:
            summary_df = self.__summary_dict[equity_tiebreaker]
            display(HTML("<h4>Tiebreaker: %s</h4>" % equity_tiebreaker))
            display(summary_df)

    def get_rank_iteration(self, assignment_df, equity_tiebreaker, iteration):
        """
        Query rank data from assigment data for a particular iteration
        :param assignment_df: assigment data
        :param equity_tiebreaker: method used as equity tiebreaker
        :param iteration: iteration
        :return:
        """
        self.augment_assigment(assignment_df, equity_tiebreaker)
        assignment_df['iteration'] = iteration
        assignment_df['method'] = equity_tiebreaker

        return assignment_df[['iteration', self.__diversity_category_col, self.__focal_label, 'cutoff_tiebreaker',
                              'rank', 'method', self.__focal_block, self.__tiebreaker_status]]
        # return assignment_df[['iteration', self.__focal_label, 'tiebreaker', 'rank', 'method']]

    def query_rank_df(self):
        """
        Aggregate rank data
        :return:
        """
        rank_results_df = []
        for equity_tiebreaker in self.__equity_tiebreaker_list:
            for iteration in range(self.__num_iterations):
                filename = self.__filename_template.format(equity_tiebreaker, iteration)
                assignment_df = pd.read_csv(self.__assignment_dir+filename).set_index('studentno')
                rank_results_df += [self.get_rank_iteration(assignment_df, equity_tiebreaker, iteration).reset_index()]

        self.__rank_results_df = pd.concat(rank_results_df, axis=0)

    def get_improvement_over_none(self, df):
        """
        Query the change of the average rank for each student over the none policy.
        :param df: rank data from `get_rank_df` method
        :return:
        """
        if "none" not in self.__equity_tiebreaker_list:
            raise Exception("`none` method is not in the loaded tiebreaker data")
        df = df.groupby(['method', 'studentno']).agg({'rank': 'mean', self.__tiebreaker_status: get_group_value, self.__diversity_category_col: get_group_value,
                                                      self.__focal_label: get_group_value})
        df_none = df.loc['none']
        df['change'] = np.nan
        for equity_tiebreaker in self.__equity_tiebreaker_list:
            if equity_tiebreaker == 'none':
                pass
            df.loc[equity_tiebreaker, 'change'] = (df.loc[equity_tiebreaker]['rank'] - df_none['rank']).values
        return df.reset_index()

    def get_improvement_tp(self, df):
        """
        Queries the average rank of each student that was correctly labeled (True Positvie) with the block tiebreaker
        for each method. This method also computes the average rank of those students with the non method. In other
        words, for each method we compute the average rank for each student with the tiebreaker and without.
        :param df: rank data from `get_rank_df` method
        :return:
        """
        if "none" not in self.__equity_tiebreaker_list:
            raise Exception("`none` method is not in the loaded tiebreaker data")
        df = df.groupby(['method', 'studentno']).agg({'rank': 'mean', self.__tiebreaker_status: get_group_value})
        df_none = df.loc['none']
        new_rows = []
        for equity_tiebreaker in self.__equity_tiebreaker_list:
            if equity_tiebreaker == 'none':
                pass
            df_eqtb = df.loc[equity_tiebreaker]
            mask = df_eqtb[self.__tiebreaker_status] == "TP"
            method_rows = df_eqtb.loc[mask, ['rank']].copy()
            method_rows['method'] = equity_tiebreaker
            method_rows['label'] = "with tiebreaker"
            none_rows = df_none.loc[method_rows.index, ['rank']].copy()
            none_rows['method'] = equity_tiebreaker
            none_rows['label'] = "without tiebreaker"

            new_rows += [method_rows.reset_index()] + [none_rows.reset_index()]
        return pd.concat(new_rows, axis=0)

    def plot_improvement_over_none(self):
        """
        Plot query from method `improvement_over_none`.
        :return:
        """
        df_change = self.get_improvement_over_none(self.__rank_results_df.copy())
        mask = df_change['method'] != "none"
        df_change = df_change.loc[mask]
        fig1, ax1 = plt.subplots(figsize=figsize)
        sns.boxplot(ax=ax1, x="method", y="change",
                    hue=self.__tiebreaker_status,
                    hue_order=self.__tiebreaker_status_ordering,
                    data=df_change,
                    showfliers = False)
        sns.despine(offset=10, trim=False)
        plt.legend(bbox_to_anchor=(.95, 1), loc=2, borderaxespad=0., title='Status')
        plt.savefig('outputs/boxplot_improvement_over_none.png', bbox_inches='tight')
        plt.show()
        
        fig2, ax2 = plt.subplots(figsize=figsize)
        sns.barplot(ax=ax2, data=df_change, x="method", hue=self.__tiebreaker_status,
                    hue_order=self.__tiebreaker_status_ordering, y="change")
        plt.legend(bbox_to_anchor=(.95, 1), loc=2, borderaxespad=0., title='Status')
        plt.savefig('outputs/barplot_improvement_over_none_method.png', bbox_inches='tight')
        plt.show()
        
        fig3, ax3 = plt.subplots(figsize=(6.8,5.2))
        sns.barplot(ax=ax3, data=df_change, x="method", hue=self.__focal_label,
                    hue_order=self.__tiebreaker_status_ordering, y="change")
        plt.legend(bbox_to_anchor=(.95, 1), loc=2, borderaxespad=0., title='Focal')
        plt.savefig('outputs/barplot_improvement_over_none_focal.png', bbox_inches='tight')
        plt.show()
        
    def plot_improvement_over_none_method(self, method):
        """
        Plot query from method `improvement_over_none` by method.
        :return:
        """
        df_change = self.get_improvement_over_none(self.__rank_results_df.copy())
        mask = df_change['method'] == method
        df_change = df_change.loc[mask]
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(ax=ax, data=df_change, x=self.__diversity_category_col,
                    hue=self.__tiebreaker_status,
                    hue_order=self.__tiebreaker_status_ordering, y="change")
        plt.legend(bbox_to_anchor=(.95, 1), loc=2, borderaxespad=0., title='Status')
        plt.savefig('outputs/plot_improvement_over_none_method_%s.png' % method, bbox_inches='tight')
        plt.show()
        

    def plot_improvement_tp(self):
        """
        Plot query from method `improvement_tp`.
        :return:
        """
        df_tp = self.get_improvement_tp(self.__rank_results_df)
        fig1, ax1 = plt.subplots(figsize=figsize)
        sns.boxplot(ax=ax1, x="method", y="rank",
                    hue="label",
                    data=df_tp,
                    showfliers = False)
        sns.despine(offset=10, trim=False)
        plt.legend(bbox_to_anchor=(.95, 1), loc=2, borderaxespad=0., title='Status')
        plt.savefig('outputs/boxplot_improvement_tp.png', bbox_inches='tight')
        plt.show()
        
        fig2, ax2 = plt.subplots(figsize=figsize)
        sns.barplot(ax=ax2, data=df_tp, x="method", hue='label', y="rank")
        plt.legend(bbox_to_anchor=(.95, 1), loc=2, borderaxespad=0., title='Status')
        plt.savefig('outputs/barplot_improvement_tp.png', bbox_inches='tight')
        plt.show()

    def rank_results_bar_plot(self, x_axis=None, hue="method"):
        """
        Generate histogram form rank results data.
        :param x_axis: x axis column
        :param hue: group column
        :return:
        """
        if x_axis is None:
            x_axis = self.__tiebreaker_status
        if hue == 'method':
            hue_order=self.__equity_tiebreaker_list
        else:
            hue_order=None
            
        plot_data = self.__rank_results_df[[x_axis, hue]].copy()
        
        if plot_data[x_axis].dtype in [np.int64, np.float64]:
            plot_data[x_axis] = plot_data[x_axis].fillna(0).apply(lambda x: "%i" % x)
        fig, ax = plt.subplots(figsize=(6.8,5.2))
        sns.histplot(ax=ax, x=x_axis, hue=hue, hue_order=hue_order, data=plot_data, 
                     multiple="dodge", shrink=.8,
                     stat="probability", common_norm=False)
        plt.savefig('outputs/rank_results_bar_plot_%s.png' %  x_axis, bbox_inches='tight')
        plt.show()
        
    def rank_results_bar_plot_method(self, method, x_axis=None, hue=None):
        """
        Generate histogram form rank results data.
        :param x_axis: x axis column
        :param hue: group column
        :return:
        """
        if x_axis is None:
            x_axis = self.__tiebreaker_status
        if hue is None:
            hue = self.__diversity_category_col
        
        mask = self.__rank_results_df['method'] == method
        plot_data = self.__rank_results_df.loc[mask, [x_axis, hue]].copy()
        
        if plot_data[x_axis].dtype in [np.int64, np.float64]:
            plot_data[x_axis] = plot_data[x_axis].fillna(0).apply(lambda x: "%i" % x)
        fig, ax = plt.subplots(figsize=(6.8,5.2))
        sns.histplot(ax=ax, x=x_axis, hue=hue, data=plot_data, multiple="dodge", shrink=.8,
                          stat="probability", common_norm=False)
        plt.savefig('outputs/rank_results_bar_plot_method_%s.png' %  x_axis, bbox_inches='tight')
        plt.show()

    def rank_results_bar_plot_by_method(self, x_axis=None, hue=None):
        """
        Generate histogram form rank results data by method.
        :param x_axis: x axis column
        :param hue: group column
        """
        if x_axis is None:
            x_axis = self.__cutoff_tiebreaker
        if hue is None:
            hue = self.__focal_label

        for equity_tiebreaker in self.__equity_tiebreaker_list:
            mask = self.__rank_results_df['method'] == equity_tiebreaker
            display(HTML("<h3>Tiebreaker: %s</h3>" % equity_tiebreaker))
            fig, ax = plt.subplots(figsize=(6.8,5.2))
            sns.histplot(ax=ax, x=x_axis, hue=hue, data=self.__rank_results_df.loc[mask], multiple="dodge", shrink=.8,
                              stat="probability", common_norm=False)
            plt.savefig('outputs/tiebreaker_distribution_%s.png' % equity_tiebreaker, bbox_inches='tight')
            plt.show()

    def regression_analysis_all(self):
        """
        Regression analysis with tiebreaker covariates and diversity category.
        :return:
        """
        for equity_tiebreaker in self.__equity_tiebreaker_list:
            self.regression_analysis_method(equity_tiebreaker)

    def regression_analysis_method(self, equity_tiebreaker):
        """
        Regression analysis with tiebreaker covariates and diversity category by method.
        :param equity_tiebreaker: method
        :return:
        """
        mask = self.__rank_results_df['method'] == equity_tiebreaker
        df = self.__rank_results_df.loc[mask]
        df = pd.concat([df, pd.get_dummies(df[self.__tiebreaker_status])], axis=1).groupby('studentno').mean()
        df = df.groupby('studentno').mean()
        y = df['rank'].copy()
        x = df.copy().drop(columns=['iteration', 'rank', self.__focal_block, 'focal'])
        x['intercept'] = 1.
        model = sm.OLS(y, x)
        results = model.fit()
        display(HTML("<h3>Tiebreaker: %s</h3>" % equity_tiebreaker))
        display(results.summary())

    def topkrank(self, k=3, hue=None):
        """
        Estimate proportion of students that get one of their top k choices. The estimation is done by averaging
        the proportion of top k rank over all iterations.
        :param k: number of options to consider
        :param hue: column to use for grouping
        :return:
        """
        if hue is None:
            hue = self.__focal_label
        
        topkrank_pct = self.__rank_results_df.groupby(['method',
                                                       hue,
                                                       'iteration']).agg({'rank': lambda x: topkrank(x, k)})
        
        
        fig1, ax1 = plt.subplots(figsize=(6.8,5.2))
        sns.barplot(ax=ax1, data=topkrank_pct.reset_index(), x="method", hue=hue, y="rank")
        for hue_val in topkrank_pct.loc['none'].index.get_level_values(hue).unique():
            plt.axhline(y=topkrank_pct.loc['none'].loc[hue_val, 'rank'].mean(), color='black')
        plt.legend(bbox_to_anchor=(.95, 1), loc=2, borderaxespad=0., title=hue)
        plt.savefig('outputs/top%irank.png' % k, bbox_inches='tight')
        plt.show()




