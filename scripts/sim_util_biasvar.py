import copy

import numpy as np
import pandas as pd


### Get predictions and results


def get_preds(params_dicts, make_model, make_X_train, X_test, f, sigma=1,
              n_iter=20):
    preds_dict_list = []

    def _get_preds_helper(model):
        preds_list = []
        for i in range(n_iter):
            X_train = make_X_train()
            y_train = make_reg_responses(X_train, f, sigma)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            preds_list.append(preds)
        return np.array(preds_list)

    for params_dict in params_dicts:
        model = make_model(params_dict)
        preds = _get_preds_helper(model)
        preds_dict = copy.deepcopy(params_dict)
        preds_dict["preds"] = preds
        preds_dict_list.append(preds_dict)
    return preds_dict_list


def get_bias(preds, X_test, f):
    preds_mean = preds.mean(axis=0)
    bias = np.mean((preds_mean - f(X_test)) ** 2)
    return bias


def get_variance(preds):
    return np.mean(np.var(preds, axis=0))


def get_expected_mse(preds, X_test, f):
    return np.mean((preds - f(X_test)) ** 2)


def get_results(preds_dict_list, X_test, f):
    results_dict_list = []
    for preds_dict in preds_dict_list:
        results_dict = dict({})
        for k, v in preds_dict.items():
            if k != "preds":
                results_dict[k] = v
        results_dict["bias"] = get_bias(preds_dict["preds"], X_test, f)
        results_dict["variance"] = get_variance(preds_dict["preds"])
        results_dict["mse"] = get_expected_mse(preds_dict["preds"], X_test, f)
        results_dict_list.append(results_dict)
    results_df = pd.DataFrame(results_dict_list)
    return results_df


### DGP

def make_gaussian_X(n, p, scale=1):
    return np.random.randn(n, p) * scale


def make_reg_responses(X, f, sigma=1):
    return f(X) + np.random.randn(X.shape[0]) * sigma


def make_clf_responses(X, f):
    return np.random.binomial(1, f(X))
