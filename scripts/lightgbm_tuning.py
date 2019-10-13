import os
import sys
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
sys.path.append(path + '/../')

from fraud.lightgbm_tuning import tune
import fraud.feature_engineering as fe
import fraud.utils as utils

import pandas as pd

# import warnings
# warnings.filterwarnings("ignore")


def baseline(train, test):
    # drop useless cols, map emails
    train, test = fe.drop_mostly_null_and_skewed_cols(train, test)
    train, test = fe.map_emails(train), fe.map_emails(test)
    train, test = fe.encode_categorical_features(train, test)

    y_test = train["isFraud"]

    train = train.drop(["TransactionID", "isFraud", "TransactionDT"], axis=1)
    test = test.drop(["TransactionID", "TransactionDT"], axis=1)

    train = train.fillna(-999)
    test = test.fillna(-999)

    return train, y_test, test


def baseline_txn_amt_dt(train, test):
    # drop useless cols, map email, txn amount and date
    train, test = fe.drop_mostly_null_and_skewed_cols(train, test)
    train, test = fe.map_emails(train), fe.map_emails(test)
    train, test = fe.map_transaction_dt(train), fe.map_transaction_dt(test)
    train, test = fe.map_transaction_amount(train, test)
    train, test = fe.encode_categorical_features(train, test)

    y_test = train["isFraud"]

    train = train.drop(["TransactionID", "isFraud", "TransactionDT"], axis=1)
    test = test.drop(["TransactionID", "TransactionDT"], axis=1)

    train = train.fillna(-999)
    test = test.fillna(-999)

    return train, y_test, test


def baseline_txn_dt(train, test):
    # drop useless cols, map email, txn date features
    train, test = fe.drop_mostly_null_and_skewed_cols(train, test)
    train, test = fe.map_emails(train), fe.map_emails(test)
    train, test = fe.map_transaction_dt(train), fe.map_transaction_dt(test)
    train, test = fe.encode_categorical_features(train, test)

    y_test = train["isFraud"]

    train = train.drop(["TransactionID", "isFraud", "TransactionDT"], axis=1)
    test = test.drop(["TransactionID", "TransactionDT"], axis=1)

    train = train.fillna(-999)
    test = test.fillna(-999)

    return train, y_test, test

bound_lgb = {
    "num_leaves": (70, 900),
    "min_child_weight": (0.001, 0.07),
    "feature_fraction": (0.1, 0.9),
    "bagging_fraction": (0.1, 0.9),
    "max_depth": (-1, 50),
    "learning_rate": (0.05, 0.9),
    "reg_alpha": (0.3, 0.9),
    "reg_lambda": (0.3, 0.9),
    "min_data_in_leaf": (50, 300),
}

# train = pd.read_pickle('../input/train.pkl')
# test = pd.read_pickle('../input/test.pkl')

# train, test = utils.load_data(reduce_mem=False)

# train, y_test, test = baseline_txn_dt(train, test)

# train, test = utils.reduce_mem_usage(train), utils.reduce_mem_usage(test)
# print("Pickling dataframes")
# train.to_pickle('train_baseline_txn_amt_dt.pkl')
# test.to_pickle('test_baseline_txn_amt_dt.pkl')

train = pd.read_pickle('../input/train_baseline_txn_dt.pkl')
test = pd.read_pickle('../input/test_baseline_txn_dt.pkl')
y_test = pd.read_pickle('../input/y_test.pkl')

target, params = tune(train, y_test, bound_lgb)

for p in ['num_leaves', 'max_depth', 'min_data_in_leaf']:
    params[p] = int(params[p])

print(target)
print(params)
