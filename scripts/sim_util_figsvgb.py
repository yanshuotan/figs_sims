import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor, \
    GradientBoostingClassifier

from imodels import FIGSRegressor, FIGSClassifier


def make_model(params_dict, regressor=True):
    if regressor:
        kw_dict = {"figs": FIGSRegressor(),
                   "gb": GradientBoostingRegressor(),
                   "cart": DecisionTreeRegressor(),
                   "linear": LinearRegression()}
    else:
        kw_dict = {"figs": FIGSClassifier(),
                   "gb": GradientBoostingClassifier(),
                   "cart": DecisionTreeClassifier(),
                   "linear": LogisticRegression()}
    model_type = params_dict["type"]
    n_rules = params_dict["n_rules"]
    model = kw_dict[model_type]
    if model_type == "figs":
        model.set_params(max_rules=n_rules)
    elif model_type == "gb":
        max_depth = params_dict["max_depth"]
        n_estimators = np.ceil(n_rules / (2 ** max_depth - 1)).astype(int)
        model.set_params(max_depth=max_depth, n_estimators=n_estimators,
                         learning_rate=1)
    elif model_type == "cart":
        model.set_params(max_leaf_nodes=(n_rules + 1))
    else:
        pass
    return model


def make_params_dicts(n_rules_grid, max_depth_grid):
    params_dicts = []
    for n_rules in n_rules_grid:
        params_dicts.append({"type": "figs",
                             "n_rules": n_rules,
                             "max_depth": None})
        params_dicts.append({"type": "cart",
                             "n_rules": n_rules,
                             "max_depth": None})
        for max_depth in max_depth_grid:
            params_dicts.append({"type": "gb",
                                 "n_rules": n_rules,
                                 "max_depth": max_depth})
    params_dicts.append({"type": "linear",
                         "n_rules": None,
                         "max_depth": None})
    return params_dicts
