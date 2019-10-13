from sklearn.model_selection import train_test_split, KFold
from bayes_opt import BayesianOptimization
import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")

def tune(train, y_test, bound_lgb, init_points=10, n_iter=15):
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
        early_stopping_rounds = 50
        xgtrain = lgb.Dataset(train_m, label=label_m)
        xgvalid = lgb.Dataset(train_val, label=label_val)
        num_boost_round = 200
        model_lgb = lgb.train(
            params,
            xgtrain,
            valid_sets=[xgtrain, xgvalid],
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=0,
        )

        score = roc_auc_score(label_val, model_lgb.predict(train_val))
        return score

    # train on the first 75% of the data and then validation on the last 25%
    idxT = train.index[:3 * len(train) // 4]
    idxV = train.index[3 * len(train) // 4:]
    train_m, train_val = train.loc[idxT], train.loc[idxV]
    label_m, label_val = y_test.loc[idxT], y_test.loc[idxV]
    
    lgb_bo = BayesianOptimization(objective, bound_lgb, random_state=42)
    print("-" * 130)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        lgb_bo.maximize(
            init_points=init_points, n_iter=n_iter, acq="ucb", xi=0.0, alpha=1e-5
        )

    target = lgb_bo.max["target"]
    params = lgb_bo.max["params"]

    return target, params
