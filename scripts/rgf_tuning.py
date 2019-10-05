"""
Random Greed Forests
--------------------

Hyper parameter tuning for FastRGF (RGF is too slow)
Works but getting poor results (approx 62 ROC)
"""

from sklearn.model_selection import train_test_split, KFold
from bayes_opt import BayesianOptimization
from rgf.sklearn import FastRGFClassifier 
import numpy as np
from sklearn.metrics import roc_auc_score
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

bound_rgf = {
    "max_depth": (1, 50),
    "max_leaf": (1000, 2000),
    "l1": (100, 300), # [0, 1000]
    "l2": (5000, 7000), # [1000, 10000]
    "min_samples_leaf": (5, 100),
    "learning_rate": (0.2, 0.9),
}

def tune(train, y_test, init_points=10, n_iter=15):
    def objective(
        max_depth,
        max_leaf,
        l1,
        l2,
        min_samples_leaf,
        learning_rate
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
            # algorithm="RGF_Sib",
            # test_interval=100,
        )
        model.fit(train_m, val1)
        score = roc_auc_score(val2, model.predict(val_m_train))
        return score
    
    train_m, val_m_train, val1, val2 = train_test_split(
        train, y_test, test_size=0.3, random_state=10, stratify=y_test
    )
    
    lgb_bo = BayesianOptimization(objective, bound_rgf, random_state=42)
    print("-" * 73)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        lgb_bo.maximize(
            init_points=init_points, n_iter=n_iter, acq="ucb", xi=0.0, alpha=1e-5
        )

    target = lgb_bo.max["target"]
    params = lgb_bo.max["params"]

    return target, params
        
train = pd.read_pickle('../input/train_baseline.pkl')
test = pd.read_pickle('../input/test_baseline.pkl')
y_test = pd.read_pickle('../input/y_test.pkl')

target, param = tune(train, y_test, 5, 5)

print(target)
print(param)