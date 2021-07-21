import sys
sys.path.append('../../')

import numpy as np
import pandas as pd
from tqdm import trange

_cache_path = '../src/d04_modeling/cache/'
_default_fname = 'value_function.pkl'

class KnapsackApprox:
    def __init__(self, eps, data: pd.DataFrame, value_col, weight_col, scale=True):
        data.sort_values(value_col, inplace=True)
        self.value_col = value_col
        self.weight_col = weight_col
        self.v = data[value_col].rename('v').to_frame() # values of each item
        self.w = data[weight_col] # weights of each item

        n = len(data)
        self.n = n

        if scale:
            v_max = self.v['v'].max()
            b = (eps / (2 * n)) * v_max
            print("Using scaling parameter b = %.4f" % b)
            self.v['v_hat'] = np.ceil(self.v['v'].values / b).astype('int64')
        else:
            # TODO: Any considerations if I don't scale the parameters and still apply the ceil function?
            # TODO: What is the approximation error in this case? (1 + eps) factor below the maximum possible.
            # TODO: What if the values are already small and integer?
            # TODO: How much can we improve by filtering out certain blocks befor hand?
            self.v['v_hat'] = np.ceil(self.v['v']).astype('int64')

        self.v['cumsum'] = self.v['v_hat'].cumsum()
        self.value_function = np.zeros((n, self.v['cumsum'].iloc[-1]+1)) + np.nan

    def solve(self):

        self.value_function[:, 0] = 0

        for i in trange(self.n):
            v_range = np.arange(1, self.v['cumsum'].iloc[i]+1)
            w_i = self.w.iloc[i]
            v_i = self.v['v_hat'].iloc[i]
            for v in v_range:
                if i == 0:
                    self.value_function[i, v] = w_i
                    continue
                if v > self.v['cumsum'].iloc[i-1]:
                    self.value_function[i, v] = w_i + self.get_values(i-1, v-v_i)
                else:
                    w1 = self.get_values(i-1, v)
                    w2 = w_i + self.get_values(i-1, max(0, v-v_i))
                    self.value_function[i, v] = min(w1, w2)

        return None

    def get_values(self, i, v):
        if i < 0:
            return 0.
        w = self.value_function[i, v]
        if np.isfinite(w):
            return w
        else:
            return 0.

    def get_solution(self, w_max):
        solution_set = []

        v_opt = np.max(np.argwhere(self.value_function[-1][:] <= w_max).flatten())
        v = v_opt
        i = self.value_function.shape[0]-1

        while v > 0:
            w_i = self.w.iloc[i]
            v_i = self.v['v_hat'].iloc[i]
            w1 = self.get_values(i, v)
            w2 = w_i + self.get_values(i-1, v-v_i)
            if w1 == w2:
                solution_set += [self.v.index[i]]
                v -= v_i
            i -= 1

        return v_opt, solution_set
    
    def get_value_per_weight(self):
        solution_weights = self.value_function[-1][:]
        results = pd.Series(solution_weights, name='weights')
        results.index.name = 'values'
        results = results.to_frame().reset_index()
        
        return results.groupby('weights').max()
    
    def save_value_function(self, fname=None):
        if fname is None:
            pd.DataFrame(self.value_function).to_pickle(_cache_path + _default_fname)
        else:
            pd.DataFrame(self.value_function).to_pickle(_cache_path + fname)
            
    def load_value_function(self, fname=None):
        if fname is None:
            df = pd.read_pickle(_cache_path + _default_fname)
        else:
            df = pd.read_pickle(_cache_path + fname)
        self.value_function = df.values


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 100  # 200 e.g. is really fine, but slower

    data = pd.read_csv("../../src/d01_data/knapsack_data.csv")

    mask = data['focalRate'] > 0.0

    solver = KnapsackApprox(eps=.5, data=data.loc[mask].copy(),
                            value_col='nFocalStudents',
                            weight_col='nOtherStudents',
                            scale=False)

    solver.solve()

    plt.imshow(pd.DataFrame(solver.value_function).fillna(0))
    plt.show()

    total_students = (data['nFocalStudents'].values + data['nOtherStudents'].values).sum()
    fp_rate = 0.1
    w_max = fp_rate * total_students
    v_opt, solution_set = solver.get_solution(w_max=w_max)
    solution_set = pd.Index(solution_set, name=data.index.name)
    results = data.loc[solution_set].sum()
    assert(results['nFocalStudents'] == v_opt)
    assert(results['nOtherStudents'] <= w_max)

