import copy

import numpy as np
import pandas as pd


def get_preds(params_dicts, make_model, make_X_train, X_test, make_y,
              n_iter=20):
    preds_dict_list = []

    def _get_preds_helper(model):
        preds_list = []
        for i in range(n_iter):
            X_train = make_X_train()
            y_train = make_y(X_train)
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


def get_bias(preds, y_test):
    preds_mean = preds.mean(axis=0)
    bias = np.mean((preds_mean - y_test) ** 2)
    return bias


def get_variance(preds):
    return np.mean(np.var(preds, axis=0))


def get_expected_mse(preds, y_test):
    return np.mean((preds - y_test) ** 2)


def get_mse_sd(preds, y_test):
    return np.std(np.mean((preds - y_test) ** 2, axis=1))


def get_results(preds_dict_list, y_test):
    results_dict_list = []
    for preds_dict in preds_dict_list:
        results_dict = dict({})
        for k, v in preds_dict.items():
            if k != "preds":
                results_dict[k] = v
        results_dict["bias"] = get_bias(preds_dict["preds"], y_test)
        results_dict["variance"] = get_variance(preds_dict["preds"])
        results_dict["mse"] = get_expected_mse(preds_dict["preds"], y_test)
        results_dict["mse_sd"] = get_mse_sd(preds_dict["preds"], y_test)
        results_dict_list.append(results_dict)
    results_df = pd.DataFrame(results_dict_list)
    return results_df