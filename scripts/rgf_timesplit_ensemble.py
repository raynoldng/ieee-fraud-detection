import os, sys

full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
sys.path.append(path + "/../")

import numpy as np, pandas as pd, os, gc
from sklearn.model_selection import GroupKFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from rgf.sklearn import RGFClassifier, FastRGFClassifier

TRAIN_LEN = 590540
TEST_LEN = 506691

test = pd.read_pickle("../processed_input/X_test_encoding_deviceinfo_email.pkl")
train = pd.read_pickle("../processed_input/X_train_encoding_deviceinfo_email.pkl")
y = pd.read_pickle("../input/y_test.pkl")
y.index = train.index

assert train.shape[0] == TRAIN_LEN
assert test.shape[0] == TEST_LEN

def train_model(X_train, y_train, params):
    l1 = params["l1"]
    l2 = params["l2"]
    learning_rate = params["learning_rate"]
    max_leaf = int(params["max_leaf"])
    max_depth = int(params["max_depth"])
    min_samples_leaf = int(params["min_samples_leaf"])

    model = FastRGFClassifier(
        max_leaf=max_leaf,
        max_depth=max_depth,
        l1=l1,
        l2=l2,
        min_samples_leaf=min_samples_leaf,
        learning_rate=learning_rate,
    )
    model.fit(X_train, y_train)

    return model


def run_model(X_train, y_train, X_val, y_val, params):
    l1 = params["l1"]
    l2 = params["l2"]
    learning_rate = params["learning_rate"]
    max_leaf = int(params["max_leaf"])
    max_depth = int(params["max_depth"])
    min_samples_leaf = int(params["min_samples_leaf"])

    model = FastRGFClassifier(
        max_leaf=max_leaf,
        max_depth=max_depth,
        l1=l1,
        l2=l2,
        min_samples_leaf=min_samples_leaf,
        learning_rate=learning_rate,
    )
    model.fit(X_train, y_train)
    pred_proba = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, pred_proba)

    return pred_proba, score


# need to convert Category types to ints
for f in train.columns:
    if str(train[f].dtype) == "category":
        train[f] = train[f].cat.codes
        test[f] = test[f].cat.codes

train = train.fillna(-999)
test = test.fillna(-999)

params = {
    "l1": 186.38900372842315,
    "min_samples_leaf": 39.804375112900715,
    "l2": 5582.458280396084,
    "max_leaf": 1292.144648535218,
    "learning_rate": 0.6282970263056656,
    "max_depth": 7.83519917195005,
}

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
    [pd.Series(train.index[-536850:]), pd.Series(back_pred)],
    axis=1
)
train_ensemble_df.columns = ["TransactionID", "isFraud"]

assert train_ensemble_df.shape == (536850, 2)
assert train_ensemble_df.loc[0, 'TransactionID'] == 3040690.0

train_ensemble_df.to_csv(path_or_buf="train_ensemble_ray_rgf.csv",index=False)

# generate test labels
model = train_model(train, y, params)
test_pred = model.predict_proba(test)[:, 1]
test_ensemble_df = pd.DataFrame({
    'TransactionID': test.index,
    'isFraud': test_pred
})

test_ensemble_df.to_csv('test_ensemble_ray_rgf.csv', index=False)