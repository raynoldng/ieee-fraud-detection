import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
import lightgbm as lgb
import itertools
from sklearn.metrics import roc_auc_score
import gc
from bayes_opt import BayesianOptimization
import warnings

import os, sys
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
sys.path.append(path + '/../')
import fraud.utils as utils
import fraud.feature_engineering as fe

warnings.filterwarnings("ignore")

train, test = utils.load_data()
train, test = fe.drop_mostly_null_and_skewed_cols(train, test)
train, test = fe.map_emails(train), fe.map_emails(test)
train, test = fe.map_transaction_amount(train), fe.map_transaction_amount(test)
train, test = fe.map_transaction_dt(train), fe.map_transaction_dt(test)

train, test = fe.encode_categorical_features(train, test)
y_test = train["isFraud"]

cols_drops = ["TransactionID", "isFraud", "TransactionDT"]
train = train.drop(cols_drops, axis=1)

# TODO more date time engineering
test = test.drop(["TransactionID", "TransactionDT"], axis=1)

train = train.fillna(-999)
test = test.fillna(-999)

## Modelling the Dataset
train_m, val_m_train, val1, val2 = train_test_split(
    train, y_test, test_size=0.3, random_state=10, stratify=y_test
)
train_m_index = train_m.index
val_m_index = val_m_train.index
val1_index = val1.index
val2_index = val2.index


def objective(
    num_leaves,
    min_child_weight,
    feature_fraction,
    bagging_fraction,
    max_depth,
    learning_rate,
    reg_alpha,
    reg_lambda,
    min_data_in_leaf,
):
    global train_m
    global y_test
    global train_m_index
    global val_m_index
    global val1, val2, val1_index, val2_index
    num_leaves = int(num_leaves)
    max_depth = int(max_depth)
    min_data_in_leaf = int(min_data_in_leaf)
    assert type(num_leaves) == int
    assert type(min_data_in_leaf) == int
    assert type(max_depth) == int
    params = {
        "num_leaves": num_leaves,
        "min_child_weight": min_child_weight,
        "feature_fraction": feature_fraction,
        "bagging_fraction": bagging_fraction,
        "min_data_in_leaf": min_data_in_leaf,
        "objective": "binary",
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "boosting_type": "gbdt",
        "bagging_seed": 11,
        "metric": "auc",
        "verbosity": -1,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "random_state": 42,
    }
    oof = np.zeros(len(train_m))
    early_stopping_rounds = 50
    xgtrain = lgb.Dataset(train_m, label=val1[val1_index])
    xgvalid = lgb.Dataset(val_m_train, label=val2[val2_index])
    num_boost_round = 200
    model_lgb = lgb.train(
        params,
        xgtrain,
        valid_sets=[xgtrain, xgvalid],
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=0,
    )
    score = roc_auc_score(val2, model_lgb.predict(val_m_train))
    return score


bound_lgb = {
    "num_leaves": (70, 600),
    "min_child_weight": (0.001, 0.07),
    "feature_fraction": (0.1, 0.9),
    "bagging_fraction": (0.1, 0.9),
    "max_depth": (-1, 50),
    "learning_rate": (0.2, 0.9),
    "reg_alpha": (0.3, 0.9),
    "reg_lambda": (0.3, 0.9),
    "min_data_in_leaf": (50, 300),
}

LGB_BO = BayesianOptimization(objective, bound_lgb, random_state=42)
init_points = 10
n_iter = 15
print("-" * 130)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    LGB_BO.maximize(
        init_points=init_points, n_iter=n_iter, acq="ucb", xi=0.0, alpha=1e-5
    )


print("target", LGB_BO.max["target"])
print("params", LGB_BO.max["params"])
