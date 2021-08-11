import pandas as pd
import numpy as np
from IPython.core.display import display, HTML
import seaborn as sns
import itertools
import statsmodels.api as sm
import matplotlib.pyplot as plt
from src.d00_utils.utils import get_group_value, show_values_on_bars

sns.set_theme(style="ticks", palette="pastel")
pd.set_option('display.max_rows', None)
plt.rcParams['font.size'] = '14'
plt.rcParams['xtick.labelsize'] = '14'
plt.rcParams['ytick.labelsize'] = '14'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'TeX Gyre Heros'
figsize = (6, 5)

fn = 'False Negative'
tn = 'True Negative'
tp = 'True Positive'
fp0 = 'False Positive'
fp1 = 'Extended Positive'


def default_is_focal(row):
    return int(row['FRL'] & row['AALPI'])


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
    __tiebreaker_status_ordering = [fn, tn, tp, fp1, fp0]

    def __init__(self, method_name_dict, is_focal=None,
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
        rank_len = student_df['r1_ranked_idschool'].dropna().apply(lambda x: len(x.strip('[]').split()))
        student_df['rank_len'] = rank_len.reindex(student_df.index)
        self.__student_df = student_df
        self.__assignment_dir = assignments_dir
        self.__method_name_dict = method_name_dict
        
    def get_student_df(self):
        return self.__student_df.copy()
    
    def get_assignment_df(self, equity_tiebreaker, iteration):
        filename = self.__filename_template.format(equity_tiebreaker, iteration)
        return pd.read_csv(self.__assignment_dir+filename).set_index('studentno')

    def set_simulation_config(self, equity_tiebreaker_list, num_iterations, policy="Medium1", guard_rails=0, utility_model=True):
        """
        Set the simulation configuration we want to evaluate.
        :param equity_tiebreaker_list: list of the methods we want to evaluate
        :param num_iterations: number of iterations from simulation
        :param policy: policy name ('Medium1' or 'Con1')
        :param guard_rails: policy includes guardrails (0: yes and -1: no)
        :param utility_model: use assignment from choice by utility model
        :return:
        """
        self.__equity_tiebreaker_list = equity_tiebreaker_list
        self.__num_iterations = num_iterations
        self.__filename_template = self.get_filename_template(policy, guard_rails, utility_model)

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
        fun = lambda x: 'with tiebreaker' if x == 1 else 'without tiebreaker'
        assignment_df[self.__focal_block] = self.__student_df[equity_tiebreaker].reindex(assignment_df.index).apply(fun)
        fun = lambda x: 'focal' if x == 1 else 'not focal' 
        assignment_df[self.__focal_label] = self.__student_df[self.__focal_label].reindex(assignment_df.index).apply(fun)
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
        mask_focal = df[self.__focal_label] == 'focal'
        mask_focal_block = df[self.__focal_block] == 'with tiebreaker'
        mask_extended_focal = df['AALPI'] | df['FRL']
        df[self.__tiebreaker_status] = ''
        df.at[mask_focal & mask_focal_block, self.__tiebreaker_status] = tp
        df.at[(~mask_focal & mask_focal_block) & mask_extended_focal, self.__tiebreaker_status] = fp1
        df.at[(~mask_focal & mask_focal_block) & ~mask_extended_focal, self.__tiebreaker_status] = fp0
        df.at[mask_focal & ~mask_focal_block, self.__tiebreaker_status] = fn
        df.at[~mask_focal & ~mask_focal_block, self.__tiebreaker_status] = tn

    def get_filename_template(self, policy, guard_rails, utility_model=True):
        """
        Query the filename template for a particular policy and guard_rails setting
        :param policy: policy form simulation ('Medium1' or 'Con1')
        :param guard_rails: guard rails option (0 or -1)
        :return:
        """
        
        if utility_model:
            return self.get_filename_template_model_choice(policy, guard_rails)
        else:
            return self.get_filename_template_real_choice(policy, guard_rails)
            
    @staticmethod
    def get_filename_template_model_choice(policy, guard_rails):
        if policy == "Con1":
            if guard_rails == 0:
                Assignment_pc3_035_CTIP1_round_merged123_policyCon1GuardRails0-prefLength07_tiesSTB_prefExtend0_iteration0.csv
                return "Assignment_{}_CTIP1_round_merged123_policyCon1GuardRails0-prefLength07_tiesSTB_prefExtend0_iteration{}.csv"
            else:
                return "Assignment_{}_CTIP1_round_merged123_policyCon1-prefLength07_tiesSTB_prefExtend0_iteration{}.csv"
        elif policy == "Medium1":
            if guard_rails == 0:
                return "Assignment_{}_CTIP1_round_merged123_policyMedium1GuardRails0-prefLength07_tiesSTB_prefExtend0_iteration{}.csv"
            else:
                return "Assignment_{}_CTIP1_round_merged123_policyMedium1-prefLength07_tiesSTB_prefExtend0_iteration{}.csv"
        raise Exception("Configuration: (policy: %s, guard_rails: %s) is not available for utility model" % (policy, guard_rails))
        
    @staticmethod
    def get_filename_template_real_choice(policy, guard_rails):
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
        raise Exception("Configuration: (policy: %s, guard_rails: %s) is not available for real preferences" % (policy, guard_rails))

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
        assignment_df['method'] = self.__method_name_dict[equity_tiebreaker]

        return assignment_df[['iteration', self.__diversity_category_col, self.__focal_label, 'cutoff_tiebreaker',
                              'rank', 'method', self.__focal_block, self.__tiebreaker_status, 'designation']]
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
        df_none = df.loc[self.__method_name_dict['none']]
        df['change'] = np.nan
        for equity_tiebreaker in self.__equity_tiebreaker_list:
            if equity_tiebreaker == 'none':
                pass
            key = self.__method_name_dict[equity_tiebreaker]
            df.loc[key, 'change'] = (df.loc[key]['rank'] - df_none['rank']).values
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
        df_none = df.loc[self.__method_name_dict['none']]
        new_rows = []
        for equity_tiebreaker in self.__equity_tiebreaker_list:
            if equity_tiebreaker == 'none':
                pass
            key = self.__method_name_dict[equity_tiebreaker]
            df_eqtb = df.loc[key]
            mask = df_eqtb[self.__tiebreaker_status] == tp
            method_rows = df_eqtb.loc[mask, ['rank']].copy()
            method_rows['method'] = self.__method_name_dict[equity_tiebreaker]
            method_rows['label'] = "with tiebreaker"
            none_rows = df_none.loc[method_rows.index, ['rank']].copy()
            none_rows['method'] = self.__method_name_dict[equity_tiebreaker]
            none_rows['label'] = "without tiebreaker"

            new_rows += [method_rows.reset_index()] + [none_rows.reset_index()]
        return pd.concat(new_rows, axis=0)

    def plot_improvement_over_none(self):
        """
        Plot query from method `improvement_over_none`.
        :return:
        """
        df_change = self.get_improvement_over_none(self.__rank_results_df.copy())
        mask = df_change['method'] != self.__method_name_dict['none']
        df_change = df_change.loc[mask]
        fig1, ax1 = plt.subplots(figsize=figsize)
        hue_order = self.__tiebreaker_status_ordering
        order = [self.__method_name_dict[key] for key in self.__equity_tiebreaker_list[1:]]
        n_status = len(self.__tiebreaker_status)
        n_focal = len(self.__focal_label)
        palette = itertools.cycle(sns.color_palette())
        status_palette = [next(palette) for i in range(n_status)]
        palette = itertools.cycle(sns.color_palette())
        focal_pallete = [next(palette) for i in range(n_focal)]
        
        sns.boxplot(ax=ax1, x="method", y="change",
                    hue=self.__tiebreaker_status,
                    hue_order=hue_order, palette=status_palette,
                    order=order,
                    data=df_change,
                    showfliers = False, zorder=2)
        sns.despine(offset=10, trim=False)
        ax1.axhline(y=0, color='k', linestyle='--', linewidth=1, zorder=1)
        plt.legend(bbox_to_anchor=(0., -.20), loc=2, borderaxespad=0., ncol=3)
        plt.savefig('outputs/boxplot_improvement_over_none.png', bbox_inches='tight')
        plt.show()
        
        # fig2, ax2 = plt.subplots(figsize=figsize)
        # sns.barplot(ax=ax2, data=df_change, x="method", hue=self.__tiebreaker_status,
        #             hue_order=hue_order, y="change", palette=status_palette,)
        # plt.legend(bbox_to_anchor=(.95, 1), loc=2, borderaxespad=0., title='Status')
        # plt.savefig('outputs/barplot_improvement_over_none_method.png', bbox_inches='tight')
        # plt.show()
        
        display(df_change.groupby(['method', self.__tiebreaker_status]).agg({'change': ['mean', 'std']}))
        
        fig3, ax3 = plt.subplots(figsize=figsize)
        sns.boxplot(ax=ax3, data=df_change, x="method", hue=self.__focal_label, y="change",
                    order=order,
                    palette=focal_pallete, showfliers = False, zorder=2)
        sns.despine(offset=10, trim=False)
        ax3.axhline(y=0, color='k', linestyle='--', linewidth=1,zorder=1)
        plt.legend(bbox_to_anchor=(0., -.20), loc=2, borderaxespad=0., ncol=3)
        plt.savefig('outputs/boxplot_improvement_over_none_focal.png', bbox_inches='tight')
        plt.show()
        
    def plot_improvement_over_none_method(self, method):
        """
        Plot query from method `improvement_over_none` by method.
        :return:
        """
        df_change = self.get_improvement_over_none(self.__rank_results_df.copy())
        mask = df_change['method'] == self.__method_name_dict[method]
        df_change = df_change.loc[mask]
        hue_order = self.__tiebreaker_status_ordering
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(ax=ax, data=df_change, x=self.__diversity_category_col,
                    hue=self.__tiebreaker_status,
                    hue_order=hue_order, y="change")
        plt.legend(bbox_to_anchor=(0., -.20), loc=2, borderaxespad=0., ncol=3)
        plt.savefig('outputs/plot_improvement_over_none_method_%s.png' % method, bbox_inches='tight')
        plt.show()
        
    def plot_improvement_tp(self):
        """
        Plot query from method `improvement_tp`.
        :return:
        """
        df_tp = self.get_improvement_tp(self.__rank_results_df)
        fig1, ax1 = plt.subplots(figsize=figsize)
        order = [self.__method_name_dict[key] for key in self.__equity_tiebreaker_list[1:]]
        sns.boxplot(ax=ax1, x="method", y="rank",
                    hue="label",
                    order=order,
                    data=df_tp,
                    showfliers = False, zorder=2)
        sns.despine(offset=10, trim=False)
        plt.legend(bbox_to_anchor=(0., -.20), loc=2, borderaxespad=0., ncol=3)
        plt.savefig('outputs/boxplot_improvement_tp.png', bbox_inches='tight')
        plt.show()
        
        fig2, ax2 = plt.subplots(figsize=figsize)
        sns.barplot(ax=ax2, data=df_tp, x="method", hue='label', y="rank", order=order)
        plt.legend(bbox_to_anchor=(0., -.20), loc=2, borderaxespad=0., ncol=3)
        plt.savefig('outputs/barplot_improvement_tp.png', bbox_inches='tight')
        plt.show()
        
    def rank_results_bar_plot(self, x_axis="method", hue=None):
        """
        Generate histogram form rank results data.
        :param x_axis: x axis column
        :param hue: group column
        :return:
        """
        if hue is None:
            hue = self.__tiebreaker_status
            
        df = self.__rank_results_df.groupby([hue, x_axis])['rank'].count().unstack(hue).fillna(0)
        self.generate_bar_plot(df, x_axis, hue)
    
    def generate_bar_plot(self, df, x_axis, hue):
        df = df / df.sum(axis=1).values[:, np.newaxis]
        
        if x_axis == 'method':
            x_axis = self.__tiebreaker_status
            order = [self.__method_name_dict[key] for key in self.__equity_tiebreaker_list]
        else:
            order = df.index
        
        if hue == self.__tiebreaker_status:
            hue_order = self.__tiebreaker_status_ordering
        else:
            hue_order= df.columns

        palette = itertools.cycle(sns.color_palette())
        num_levels = len(hue_order)
        df = df.reindex(order)
        fig1, ax1 = plt.subplots(figsize=figsize)
        bar_width = 0.85
        r = range(len(order))
        bottom = np.zeros(len(order))
        for level in range(num_levels):
            key = hue_order[level]
            top = df[key]
            if key == 0:
                ax1.bar(r, top, color=next(palette), edgecolor='white', width=bar_width, label=key)
            else:
                ax1.bar(r, top, bottom=bottom, color=next(palette), edgecolor='white', 
                        width=bar_width, label=key)
            bottom += top

        # Custom x axis
        show_values_on_bars(ax1)
        ax1.set_xticks(r)
        ax1.set_xticklabels(order)
        ax1.set_xlabel(x_axis)
        ax1.set_ylabel('Proportion')
        plt.legend(bbox_to_anchor=(0., -.20), loc=2, borderaxespad=0., ncol=3)
        plt.savefig('outputs/rank_results_bar_plot_%s.png' %  x_axis, bbox_inches='tight')
        # Show graphic
        plt.show()
        

    def rank_results_bar_plot_(self, x_axis=None, hue="method"):
        """
        Generate histogram form rank results data.
        :param x_axis: x axis column
        :param hue: group column
        :return:
        """
        if x_axis is None:
            x_axis = self.__tiebreaker_status
            order = self.__tiebreaker_status_ordering
        else:
            order = None
        if hue == 'method':
            hue_order = [self.__method_name_dict[key] for key in self.__equity_tiebreaker_list]
        elif hue == 'status':
            hue_order = self.__tiebreaker_status_ordering
        else:
            hue_order=None
            
        plot_data = self.__rank_results_df[[x_axis, hue]].copy()
        
        if plot_data[x_axis].dtype in [np.int64, np.float64]:
            plot_data[x_axis] = plot_data[x_axis].fillna(0).apply(lambda x: "%i" % x)
        fig1, ax1 = plt.subplots(figsize=figsize)
        sns.histplot(ax=ax1, y=x_axis, hue=hue, hue_order=hue_order, data=plot_data, 
                     multiple="dodge", 
                     shrink=.8,
                     stat="probability", common_norm=False)
        ax1.set_xlim(0, 1.0)
        ax1.set_ylabel("")
        show_values_on_bars(ax1, h_v="h")
        plt.savefig('outputs/rank_results_bar_plot_%s.png' %  x_axis, bbox_inches='tight')
        plt.show()
        
    def rank_results_bar_plot_method(self, method, x_axis=None, hue=None):
        """
        Generate histogram form rank results data.
        :param x_axis: x axis column
        :param hue: group column
        :return:
        """
        hue_order = None
        order = None
        if x_axis is None:
            x_axis = self.__tiebreaker_status
        elif x_axis == self.__diversity_category_col:
            order = list(range(3))
        if hue is None:
            hue = self.__diversity_category_col
            hue_order = list(range(3))
        elif hue == 'status':
            hue_order = self.__tiebreaker_status_ordering
        
        mask = self.__rank_results_df['method'] == self.__method_name_dict[method]
        plot_data = self.__rank_results_df.loc[mask, [x_axis, hue]].copy()
        
        if plot_data[x_axis].dtype in [np.int64, np.float64]:
            plot_data[x_axis] = plot_data[x_axis].fillna(0).apply(lambda x: "%i" % x)
        fig, ax = plt.subplots(figsize=figsize)
        sns.histplot(ax=ax, x=x_axis, hue=hue, hue_order=hue_order, data=plot_data, multiple="dodge", shrink=.8,
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
            fig, ax = plt.subplots(figsize=figsize)
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
        df = pd.concat([df, pd.get_dummies(df[self.__tiebreaker_status])], axis=1)
        df = df.groupby('studentno').mean()
        df['rank_len'] = self.__student_df['rank_len'].reindex(df.index)
        df.dropna(inplace=True)
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
        
        
        fig1, ax1 = plt.subplots(figsize=figsize)
        order = [self.__method_name_dict[key] for key in self.__equity_tiebreaker_list]
        sns.barplot(ax=ax1, data=topkrank_pct.reset_index(), x="method", hue=hue, y="rank",
                    order=order, zorder=2)
        ax1.set_ylim(0, 1)
        ax1.set_ylabel("Proportion of students getting\n their top %i choice" % k)
        key = self.__method_name_dict['none']
        for hue_val in topkrank_pct.loc[key].index.get_level_values(hue).unique():
            plt.axhline(y=topkrank_pct.loc[key].loc[hue_val, 'rank'].mean(), color='k', linestyle='--', linewidth=1,zorder=1)
        plt.legend(bbox_to_anchor=(.95, 1), loc=2, borderaxespad=0.)
        plt.savefig('outputs/top%irank.png' % k, bbox_inches='tight')
        plt.show()
        
        display(topkrank_pct.groupby(['method', hue]).agg({'rank': ['mean', 'std']}))
        
    def rank_results_bar_plot_by_method2(self, method, x_axis="method", hue=None):
        """
        Generate histogram form rank results data.
        :param method: methodology to plot
        :param x_axis: x axis column
        :param hue: group column
        :return:
        """
        if hue is None:
            hue = self.__tiebreaker_status
            
        df = self.__rank_results_df.groupby(['method', hue, x_axis])['rank'].count().unstack(hue).fillna(0)
        key = self.__method_name_dict[method]
        self.generate_bar_plot(df.loc[key], x_axis, hue)




