import numpy as np, pandas as pd, os, gc
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from rgf.sklearn import RGFClassifier, FastRGFClassifier
from bayes_opt import BayesianOptimization
from sklearn.metrics import roc_auc_score
import warnings

import sys
sys.path.append('/home/raynoldng/Desktop/ieee-fraud-detection')
sys.path.append('/home/raynoldng/Desktop/ieee-fraud-detection/fraud')

train = pd.read_pickle('../processed_input/X_train_encoding_deviceinfo_email.pkl')
test = pd.read_pickle('../processed_input/X_test_encoding_deviceinfo_email.pkl')
y_train = pd.read_pickle('../input/y_test.pkl')

y_train.index = train.index

# need to convert Category types to ints
for f in train.columns:
    if str(train[f].dtype) == 'category':
        train[f] = train[f].cat.codes
        test[f] = test[f].cat.codes

train = train.fillna(-999)
test = test.fillna(-999)

fast_bound_rgf = {
    "max_depth": (1, 50),
    "max_leaf": (1000, 2000),
    "l1": (100, 300), # [0, 1000]
    "l2": (5000, 7000), # [1000, 10000]
    "min_samples_leaf": (5, 100),
    "learning_rate": (0.2, 0.9),
}

bound_rgf = {
    'max_leaf': (1000, 10000),
    'l2': (0.01, 0.9),
    "min_samples_leaf": (5, 300),
    "learning_rate": (0.2, 0.9),
}

def tune(train, y_test, fast, init_points=10, n_iter=15):

    def fast_objective(
        max_depth,
        max_leaf,
        l1,
        l2,
        min_samples_leaf,
        learning_rate,
    ):
        max_leaf = int(max_leaf)
        max_depth = int(max_depth)
        min_samples_leaf = int(min_samples_leaf)
        
        assert type(max_leaf) == int
        assert type(max_depth) == int
        assert type(min_samples_leaf) == int
                
        model = FastRGFClassifier(
            max_leaf=max_leaf,
            max_depth=max_depth,
            l1=l1,
            l2=l2,
            min_samples_leaf=min_samples_leaf,
            learning_rate=learning_rate,
        )
        model.fit(train_m, label_m)
        pred_proba = model.predict_proba(train_val)
        score = roc_auc_score(label_val, pred_proba[:, 1])
        return score


    def objective(
        max_leaf,
        l2,
        min_samples_leaf,
        learning_rate
    ):
        max_leaf = int(max_leaf)
        min_samples_leaf = int(min_samples_leaf)
        
        assert type(max_leaf) == int
        assert type(min_samples_leaf) == int
                
        model = RGFClassifier(
            max_leaf=max_leaf,
            l2=l2,
            min_samples_leaf=min_samples_leaf,
            learning_rate=learning_rate,
            algorithm="RGF_Sib",
            test_interval=100,
        )
        model.fit(train_m, label_m)
        pred_proba = model.predict_proba(train_val)
        score = roc_auc_score(label_val, pred_proba[:, 1])
        return score
    
    idxT = train.index[:3 * len(train) // 4]
    idxV = train.index[3 * len(train) // 4:]
    train_m, train_val = train.loc[idxT], train.loc[idxV]
    label_m, label_val = y_test.loc[idxT], y_test.loc[idxV]
    
    objective_func = fast_objective if fast else objective
    space = fast_bound_rgf if fast else bound_rgf
    lgb_bo = BayesianOptimization(objective_func, space, random_state=42)
    print("-" * 73)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        lgb_bo.maximize(
            init_points=init_points, n_iter=n_iter, acq="ucb", xi=0.0, alpha=1e-5
        )

    target = lgb_bo.max["target"]
    params = lgb_bo.max["params"]

    return target, params


target, params = tune(train, y_train, fast=True, init_points=5, n_iter=5)

print(target)
print(params)