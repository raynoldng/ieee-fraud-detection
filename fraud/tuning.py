#############################################
# HyperOpt and XGBoost
#############################################
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import make_scorer

## Hyperopt modules
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
from functools import partial
import time

import gc

# Space that we are trying to optimize
SPACE = {
    # The maximum depth of a tree, same as GBM.
    # Used to control over-fitting as higher depth will allow model
    # to learn relations very specific to a particular sample.
    # Should be tuned using CV.
    # Typical values: 3-10
    "max_depth": hp.quniform("max_depth", 7, 23, 1),
    # reg_alpha: L1 regularization term. L1 regularization encourages sparsity
    # (meaning pulling weights to 0). It can be more useful when the objective
    # is logistic regression since you might need help with feature selection.
    "reg_alpha": hp.uniform("reg_alpha", 0.01, 0.4),
    # reg_lambda: L2 regularization term. L2 encourages smaller weights, this
    # approach can be more useful in tree-models where zeroing
    # features might not make much sense.
    "reg_lambda": hp.uniform("reg_lambda", 0.01, 0.4),
    # eta: Analogous to learning rate in GBM
    # Makes the model more robust by shrinking the weights on each step
    # Typical final values to be used: 0.01-0.2
    "learning_rate": hp.uniform("learning_rate", 0.01, 0.2),
    # colsample_bytree: Similar to max_features in GBM. Denotes the
    # fraction of columns to be randomly samples for each tree.
    # Typical values: 0.5-1
    "colsample_bytree": hp.uniform("colsample_bytree", 0.3, 0.9),
    # A node is split only when the resulting split gives a positive
    # reduction in the loss function. Gamma specifies the
    # minimum loss reduction required to make a split.
    # Makes the algorithm conservative. The values can vary depending on the loss function and should be tuned.
    "gamma": hp.uniform("gamma", 0.01, 0.7),
    # more increases accuracy, but may lead to overfitting.
    # num_leaves: the number of leaf nodes to use. Having a large number
    # of leaves will improve accuracy, but will also lead to overfitting.
    "num_leaves": hp.choice("num_leaves", list(range(20, 250, 10))),
    # specifies the minimum samples per leaf node.
    # the minimum number of samples (data) to group into a leaf.
    # The parameter can greatly assist with overfitting: larger sample
    # sizes per leaf will reduce overfitting (but may lead to under-fitting).
    "min_child_samples": hp.choice("min_child_samples", list(range(100, 250, 10))),
    # subsample: represents a fraction of the rows (observations) to be
    # considered when building each subtree. Tianqi Chen and Carlos Guestrin
    # in their paper A Scalable Tree Boosting System recommend
    "subsample": hp.choice("subsample", [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    # randomly select a fraction of the features.
    # feature_fraction: controls the subsampling of features used
    # for training (as opposed to subsampling the actual training data in
    # the case of bagging). Smaller fractions reduce overfitting.
    "feature_fraction": hp.uniform("feature_fraction", 0.4, 0.8),
    # randomly bag or subsample training data.
    "bagging_fraction": hp.uniform("bagging_fraction", 0.4, 0.9)
    # bagging_fraction and bagging_freq: enables bagging (subsampling)
    # of the training data. Both values need to be set for bagging to be used.
    # The frequency controls how often (iteration) bagging is used. Smaller
    # fractions and frequencies reduce overfitting.
}


def objective(X_train, y_train, folds=7):
    def __objective(params):
        time1 = time.time()
        params = {
            "max_depth": int(params["max_depth"]),
            "gamma": "{:.3f}".format(params["gamma"]),
            "subsample": "{:.2f}".format(params["subsample"]),
            "reg_alpha": "{:.3f}".format(params["reg_alpha"]),
            "reg_lambda": "{:.3f}".format(params["reg_lambda"]),
            "learning_rate": "{:.3f}".format(params["learning_rate"]),
            "num_leaves": "{:.3f}".format(params["num_leaves"]),
            "colsample_bytree": "{:.3f}".format(params["colsample_bytree"]),
            "min_child_samples": "{:.3f}".format(params["min_child_samples"]),
            "feature_fraction": "{:.3f}".format(params["feature_fraction"]),
            "bagging_fraction": "{:.3f}".format(params["bagging_fraction"]),
        }

        print("\n############## New Run ################")
        print(f"params = {params}")
        count = 1

        tss = TimeSeriesSplit(n_splits=folds)

        score_mean = 0
        for tr_idx, val_idx in tss.split(X_train, y_train):
            clf = xgb.XGBClassifier(
                n_estimators=600,
                random_state=4,
                verbose=True,
                # tree_method="hist",  # my desktop does not support CUDA :(
                tree_method='gpu_hist',
                predictor='gpu_predictor',
                nthread=8,
                **params,
            )

            X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]
            y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            clf.fit(X_tr, y_tr)

            # adding needs_proba=True leads to a bad shape error
            score = make_scorer(roc_auc_score)(clf, X_vl, y_vl)
            # plt.show()
            score_mean += score
            print(f"{count} CV - score: {round(score, 4)}")
            count += 1
        time2 = time.time() - time1
        print(f"Total Time Run: {round(time2 / 60,2)}")
        gc.collect()
        print(f"Mean ROC_AUC: {score_mean / folds}")
        del X_tr, X_vl, y_tr, y_vl, clf, score
        return -(score_mean / folds)

    return __objective

def optimize_hyper_parameters(X_train, y_train, num_runs=15):
    # use hyper opt to decide what hyper parameters to use
    best = fmin(
        fn=objective(X_train, y_train),
        space=SPACE,
        algo=tpe.suggest,
        max_evals=num_runs,
    )
    best_params = space_eval(SPACE, best)

    return best_params
