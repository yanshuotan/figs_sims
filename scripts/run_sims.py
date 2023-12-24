import warnings
from functools import partial

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from sim_util_biasvar import get_preds, get_results
from sim_util_models import make_model, make_params_dicts
from sim_util_dgp import linear_model, lss_model, hierarchical_poly, \
    sample_normal_X


def run_sim(n_rules_grid, max_depth_grid, make_model, true_y_model,
            n=500, p=20, n_iter=20, **true_y_model_params):
    make_y = partial(true_y_model, **true_y_model_params)
    X_test = sample_normal_X(n=500, d=p)
    true_y_model_params.pop("heritability")
    y_test = true_y_model(X_test, **true_y_model_params)

    def make_X_train():
        return sample_normal_X(n=n, d=p)

    params_dicts = make_params_dicts(n_rules_grid, max_depth_grid)
    preds_dict_list = get_preds(params_dicts, make_model, make_X_train, X_test,
                                make_y, n_iter)
    results_df = get_results(preds_dict_list, y_test)
    return results_df


warnings.filterwarnings("ignore")
# Grid of algorithmic parameters
n_rules_grid = np.arange(1, 10) * 3
max_depth_grid = [1, 2]

# Grids for DGP parameters
sims_list = ["linear", "lss", "poly", "hierarchical_lss"]
param_grids = {
    "linear": {
        "heritability": [0.4],
        "s": [6],
        "sigma": [0],
        "beta": [1]
    },
    "lss": {
        "heritability": [0.2, 0.4, 0.6],
        "m": [1, 3, 5],
        "r": [2, 4],
        "sigma": [0],
        "tau": [0],
        "beta": [1]
    },
    "poly": {
        "heritability": [0.2, 0.4, 0.6],
        "m": [1, 2],
        "r": [2, 3],
        "sigma": [0],
        "beta": [1]
    },
    "hierarchical_lss": {
        "heritability": [0.2, 0.4, 0.6],
        "m": [3, 5],
        "r": [2, 4],
        "sigma": [0],
        "beta": [1],
        "lss": [True]
    }
}
true_y_model_dict = {
    "linear": linear_model,
    "lss": lss_model,
    "poly": hierarchical_poly,
    "hierarchical_lss": hierarchical_poly
}

for sim in ["linear"]:
    param_grid = param_grids[sim]
    param_combinations = list(ParameterGrid(param_grid))
    results_list = []
    for comb in tqdm(param_combinations):
        results = dict({})
        for k, v in comb.items():
            if len(param_grid[k]) > 1:
                results[k] = v
            results["vals"] = run_sim(n_rules_grid, max_depth_grid, make_model,
                                      p=100,
                                      true_y_model=true_y_model_dict[sim],
                                      n_iter=50, **comb)
        results_list.append(results)
    with open(f"results/high_dim_results_linear.pkl", "wb") as file:
        pickle.dump(results_list, file)