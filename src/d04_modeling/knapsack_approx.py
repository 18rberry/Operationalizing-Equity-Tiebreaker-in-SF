import sys
sys.path.append('../..')

import numpy as np
import pandas as pd
from tqdm import trange


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
                    self.value_function[i, v] = w_i + self.value_function[i-1, v]
                else:
                    w1 = self.value_function[i-1, v]
                    w2 = w_i + self.value_function[i, max(0, v-v_i)]
                    self.value_function[i, v] = min(w1, w2)

        return None

    def get_solution(self, w_max):
        solution_set = []

        feasible_solutions = np.argwhere(self.value_function <= w_max)

        max_solution = np.argmax(feasible_solutions[:, 1])
        i, v = feasible_solutions[max_solution]

        while i >= 0:
            w_i = self.w.iloc[i]
            v_i = self.v['v_hat'].iloc[i]

        raise NotImplementedError()


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

    plt.imshow(solver.value_function)
    plt.show()
