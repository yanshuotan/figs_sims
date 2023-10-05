import warnings

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from sim_util_biasvar import get_preds, get_results, make_gaussian_X
from sim_util_figsvgb import make_model, make_params_dicts
from sim_util_tiffany import lss_model


def run_lss_sim(n_rules_grid, max_depth_grid, make_model,
                n=100, p=10, m=1, r=2, sigma=0.1, n_iter=20):

    def f(x):
        return lss_model(x, 0, m, r, tau=0, beta=10)
    X_test = make_gaussian_X(200, p)
    def make_X_train():
        return make_gaussian_X(n, p)
    params_dicts = make_params_dicts(n_rules_grid, max_depth_grid)
    preds_dict_list = get_preds(params_dicts, make_model, make_X_train, X_test,
                                f, sigma, n_iter)
    results_df = get_results(preds_dict_list, X_test, f)
    return results_df


warnings.filterwarnings("ignore")
n_rules_grid = np.arange(1, 10) * 3
max_depth_grid = [1, 2]

param_grid = {
    "sigma": [0.1, 1, 10],
    "m": [1, 2, 3],
    "r": [1, 2, 3]
}

param_combinations = list(ParameterGrid(param_grid))

results_list = []
for comb in tqdm(param_combinations):
    results = dict({})
    sigma = comb["sigma"]
    m = comb["m"]
    r = comb["r"]
    results["vals"] = run_lss_sim(n_rules_grid, max_depth_grid,
                                  make_model, m=m, r=r, sigma=sigma,
                                  n_iter=50)
    results["sigma"] = sigma
    results["m"] = m
    results["r"] = r
    results_list.append(results)

with open("../results/lss_sim_results.pkl", "wb") as file:
    pickle.dump(results_list, file)