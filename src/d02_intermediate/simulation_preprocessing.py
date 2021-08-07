import pandas as pd
import numpy as np
import os

from src.d00_utils.utils import add_bayesian_bernoulli
from src.d02_intermediate.classifier_data_api import ClassifierDataApi
from src.d00_utils.utils import get_label

_rand_expected = "0.860182"
_aalpi_ethnicity_list = ['Black or African American', 'Hispanic/Latino', 'Pacific Islander']


class SimulationPreprocessing:
    """
    This class add a column of new tiebreaker (augment) to the student data files used for the simulation.
    """
    __recalculate = False

    def __init__(self, frl_key='tk5', period="1819",
                 output_path="/share/data/school_choice_equity/simulator_data/student/",
                 student_data_file="/share/data/school_choice/Data/Cleaned/drop_optout_{}.csv"):
        """
        :param frl_key: string that identifies which FRL data should be loaded ('tk5' or 'tk12')
        :param period: period of student data to simulate
        :param output_path: path from where to export the augmented student data
        :param student_data_file: path from where to import the raw student data
        """
        self.frl_key = frl_key
        self.output_path = output_path
        self.fname = self.get_file_name(period=period)
        student_df = pd.read_csv(student_data_file.format(period)).set_index('studentno')
        mask = student_df['grade'] == 'KG'
        self.__student_df = student_df.loc[mask]
        
    def set_recalculate(self, recalculate):
        self.__recalculate = recalculate

    @staticmethod
    def frl_cond_aalpi(row):
        """
        Compute the probability of being FRL conditional on being AALPI.
        :param row: row from FRL block data
        :return:
        """
        return row['probBoth'] / row['probAALPI']

    @staticmethod
    def frl_cond_naalpi(row):
        """
        Compute the probability of being FRL conditional on NOT being AALPI.
        :param row: row from block FRL data
        :return:
        """
        return (row['probFRL'] - row['probBoth']) / (1. - row['probAALPI'])

    @staticmethod
    def is_frl(row, frl_df: pd.DataFrame):
        """
        Simulate if each student is FRL by using the probabilities of being FRL conditional on their AALPI status.
        :param row: row from student data
        :param frl_df: blocl FRL data
        :return:
        """
        geoid = row['census_block']
        if np.isnan(geoid):
            return 0
        geoid = int(geoid)
        u = np.random.random()
        if geoid in frl_df.index:
            probs = frl_df.loc[geoid]
            if row['AALPI'] == 1:
                if u <= probs['probCondAALPI']:
                    return 1
                else:
                    return 0
            else:
                if u <= probs['probCondNAALPI']:
                    return 1
                else:
                    return 0
        else:
            return 0

    def add_frl_labels(self):
        """
        Add FRL labels to the student data
        :return:
        """
        if 'FRL' in self.__student_df.columns:
            return None
        np.random.seed(1992)
        cda = ClassifierDataApi()
        frl_df = cda.get_frl_data(frl_key=self.frl_key)
        frl_df = add_bayesian_bernoulli(frl_df)

        frl_df['probCondAALPI'] = frl_df.apply(lambda row: self.frl_cond_aalpi(row), axis=1)
        frl_df['probCondNAALPI'] = frl_df.apply(lambda row: self.frl_cond_naalpi(row), axis=1)

        self.__student_df['AALPI'] = self.__student_df['resolved_ethnicity'].isin(_aalpi_ethnicity_list).astype('int64')
        rand_test = "%.6f" % np.random.random()
        try:
            assert rand_test == _rand_expected
        except AssertionError as err:
            raise Exception("Random number generator is off %s <> %s" % (_rand_expected, rand_test))
        self.__student_df['FRL'] = self.__student_df.apply(lambda row: self.is_frl(row, frl_df), axis=1)

    def add_equity_tiebreaker(self, model, params, tiebreaker, block_indx=None):
        """
        Add equity tiebreaker columns
        :param model: focal block classifier model
        :param params: parameters of focal block classifier model
        :param tiebreaker: name of tiebreaker column
        :param block_indx: array with the block index (work around in special cases)
        :return:
        """
        solution = model.get_solution_set(params)
        print(solution)
        self.__student_df[tiebreaker] = self.__student_df['census_block'].apply(lambda x: get_label(x, solution, block_indx))
        print("Ratio of students recieving the wquity tiebreaker: %.2f" % self.__student_df[tiebreaker].mean())

    @staticmethod
    def get_file_name(period):
        """
        query file name for particular period
        :param period: student data period
        :return:
        """
        return "drop_optout_{}.csv".format(period)

    def check_consistency(self, student_out: pd.DataFrame):
        """
        Make sure the new student dataframe  is consistent with the dataframe already saved
        :param student_out: old student data
        :return:
        """
        test1 = student_out['FRL'] != self.__student_df['FRL']
        if test1.any():
            return False, "FRL values are different"
        test2 = student_out.index.difference(self.__student_df.index)
        if len(test2) > 0:
            return False, "New student data is missing students"
        test3 = self.__student_df.index.difference(student_out.index)
        if len(test3) > 0:
            return False, "New student data has additional students"

        return True, ""

    def update_student_data(self, tiebreaker):
        """
        Add tiebreaker column to old (already generated) augmented student data. In the case that there is no old
        augmented student data then this file is created for the first time.
        :param tiebreaker: name of tiebreaker column to update
        :return:
        """
        if os.path.isfile(self.output_path + self.fname):
            print("Loading student data from:\n %s" % (self.output_path + self.fname))
            student_out = pd.read_csv(self.output_path + self.fname).set_index('studentno')
            if not self.__recalculate:
                test, flag = self.check_consistency(student_out)
                if not test:
                    self.__student_df['FRL'] = student_out['FRL'].reindex(self.__student_df.index)
                test, flag = self.check_consistency(student_out)
                if not test:
                    raise Exception("Consistency Error: %s" % flag)
            if tiebreaker not in student_out.columns or self.__recalculate:
                print("Updating %s in student data..." % tiebreaker)
                student_out[tiebreaker] = self.__student_df[tiebreaker]
                self.__recalculate = False
            else:
                raise Exception("Tiebreaker already exists!")

        else:
            print("Creating student data:\n %s" % (self.output_path + self.fname))
            student_out = self.__student_df.copy()

        return student_out

    def save_student_data(self, student_out):
        """
        Export augmented student data
        :param student_out: augmented student data
        :return:
        """
        print("Saving to:\n  %s" % (self.output_path + self.fname))
        if student_out.index.name == 'studentno':
            student_out.reset_index(inplace=True)
            student_out.set_index('Unnamed: 0', inplace=True)
            student_out.reset_index(inplace=True)
        student_out.to_csv(self.output_path + self.fname, index=False)
