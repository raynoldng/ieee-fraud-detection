########################################################
# Training of Lightgbm model after tuning
########################################################
import os, sys

full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
sys.path.append(path + "/../")
from fraud.utils import load_data
import fraud.feature_engineering as fe
from fraud.lightgbm_tuning import tune
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

params = {
    "bagging_fraction": 0.3941825796609002,
    "feature_fraction": 0.3846040853732473,
    "learning_rate": 0.22910767154232378,
    "max_depth": 49.36834392450468,
    "min_child_weight": 0.02501370683124294,
    "min_data_in_leaf": 299.5652731433522,
    "num_leaves": 321.3501221967029,
    "reg_alpha": 0.7735276016066162,
    "reg_lambda": 0.5664371765586673,
}
num_boost_round = 1000

train, test = load_data()

train, test = fe.drop_mostly_null_and_skewed_cols(train, test)
train, test = fe.map_emails(train), fe.map_emails(test)
train, test = fe.map_transaction_amount(train), fe.map_transaction_amount(test)
train, test = fe.map_transaction_df(train), fe.map_transaction_df(test)
train, test = fe.encode_categorical_features(train, test)

y_test = train["isFraud"]

d_train = lgb.Dataset(train, label=y_test)
d_train.save_binary("baseline_txn_amt_dt")
print("saving data binary")

model = lgb.train(params, d_train, verbose_eval=False, num_boost_round=1000)

print("Saving model...")
model.save_model("lightgbm_baseline_txn_amt_df.txt")
print("Model saved to model.txt")

