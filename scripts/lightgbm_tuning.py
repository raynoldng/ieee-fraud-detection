import warnings

import os, sys
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
sys.path.append(path + '/../')
import fraud.utils as utils
import fraud.feature_engineering as fe
from fraud.lightgbm_tuning import tune

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

target, params = tune(train, y_test)
