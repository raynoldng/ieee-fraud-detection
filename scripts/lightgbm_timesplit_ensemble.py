import numpy as np, pandas as pd, os, gc
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

import warnings

warnings.filterwarnings("ignore")

test = pd.read_pickle("../processed_input/X_test_encoding_deviceinfo_email.pkl")
train = pd.read_pickle("../processed_input/X_train_encoding_deviceinfo_email.pkl")
y = pd.read_pickle("../input/y_test.pkl")
y.index = train.index

assert train.shape[0] == 590540
assert test.shape[0] == 506691


params = {
    "bagging_fraction": 0.51138755073088926,
    "feature_fraction": 0.57393165508963395,
    "learning_rate": 0.065644111129998017,
    "max_depth": 36.060235965987744,
    "min_child_weight": 0.012766164534423117,
    "min_data_in_leaf": 66.262898246319878,
    "num_leaves": 1142.2406570962667,
    "reg_alpha": 0.87937921984473566,
    "reg_lambda": 0.78503840886987675,
}


def run_model(X_train, y_train, X_val, y_val, params):
    params["max_depth"] = int(params["max_depth"])
    params["min_data_in_leaf"] = int(params["min_data_in_leaf"])
    params["num_leaves"] = int(params["num_leaves"])
    params["metric"] = "auc"

    d_train = lgb.Dataset(X_train, label=y_train)
    d_val = lgb.Dataset(X_val, label=y_val, reference=d_train)

    model = lgb.train(
        params,
        d_train,
        verbose_eval=-1,
        num_boost_round=1000,
        early_stopping_rounds=20,
        valid_sets=d_val,
    )
    pred_proba = model.predict(X_val)
    score = roc_auc_score(y_val, pred_proba)
    print(score)

    return pred_proba, score


tscv = TimeSeriesSplit(n_splits=10)
model = None
preds = []
scores = []

for train_index, val_index in [i for i in tscv.split(train)]:
    print(len(train_index), len(val_index))
    X_train, X_val = train.iloc[train_index, :].copy(), train.iloc[val_index, :].copy()
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    pred, score = run_model(X_train, y_train, X_val, y_val, params)

    preds.append(pred)
    scores.append(score)

back_pred = []
for pred in preds:
    back_pred.extend(pred)

train_ensemble_df = pd.concat(
    [X_train[-536850:].reset_index().TransactionID, pd.Series(back_pred)], axis=1
)
train_ensemble_df.columns = ["TransactionID", "isFraud"]
train_ensemble_df.to_csv(path_or_buf="train_ensemble_ray_lightgbm.csv", index=False)
print("Done")
